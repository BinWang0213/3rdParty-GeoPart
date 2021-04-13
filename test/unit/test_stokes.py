import pytest

import numpy as np

import geopart.stokes
import geopart.stokes.incompressible
import geopart.stokes.compressible
import dolfin
import dolfin_dg as dg
from ufl import sym, grad, Identity, div, dx, outer, tr


class MMS(geopart.stokes.StokesModel):
    def __init__(self, comm, nx, ny):
        self.mesh = dolfin.RectangleMesh.create(
            comm, [dolfin.Point(-1.0, -1.0), dolfin.Point(1.0, 1.0)],
            [nx, ny], dolfin.CellType.Type.triangle, "left/right")

        u_soln_code = ("-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])",
                       "x[1] * sin(x[1]) * exp(x[0])")
        p_soln_code = "2.0 * exp(x[0]) * sin(x[1]) " \
                      "+ 1.5797803888225995912 / 3.0"

        u_soln = dolfin.Expression(u_soln_code, degree=4, domain=self.mesh)
        p_soln = dolfin.Expression(p_soln_code, degree=4, domain=self.mesh)
        f = -div(2 * sym(grad(u_soln)) - p_soln * Identity(2))

        self.u_soln = u_soln
        self.p_soln = p_soln

        super().__init__(eta=1, f=f)

    @staticmethod
    def generate_facet_function(mesh):
        ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        dolfin.CompiledSubDomain(
            "near(x[0], -1.0) or near(x[1], -1.0)").mark(ff, 1)
        dolfin.CompiledSubDomain(
            "near(x[0], 1.0) or near(x[1], 1.0)").mark(ff, 2)
        return ff

    def generate_bcs(self, ds):
        n = dolfin.FacetNormal(self.mesh)
        u_soln, p_soln = self.u_soln, self.p_soln
        gN = (2*sym(grad(u_soln))
              - p_soln*Identity(self.mesh.geometry().dim())) * n
        return [dg.DGDirichletBC(ds(1), u_soln),
                dg.DGNeumannBC(ds(2), gN)]


elements_rates = [
    (geopart.stokes.incompressible.TaylorHood, 3, 2, 2, 1),
    (geopart.stokes.incompressible.HDG2, 3, 2, 2, 1),
    (geopart.stokes.incompressible.HDG, 2, 1, 1, None),
]


@pytest.mark.parametrize("run_data", elements_rates)
def test_stokes_convergence(run_data):
    element_class, ul2_rate, uh1_rate, pl2_rate, ph1_rate = run_data

    n_eles = [8, 16, 32]
    erroru_l2 = np.array([0.0] * len(n_eles), dtype=np.double)
    erroru_h1 = np.array([0.0] * len(n_eles), dtype=np.double)
    erroru_div = np.array([0.0] * len(n_eles), dtype=np.double)
    errorp_l2 = np.array([0.0] * len(n_eles), dtype=np.double)
    errorp_h1 = np.array([0.0] * len(n_eles), dtype=np.double)
    hsizes = np.array([0.0] * len(n_eles), dtype=np.double)

    for j, n_ele in enumerate(n_eles):
        element_cls = element_class()
        model = MMS(dolfin.MPI.comm_world, n_ele, n_ele)
        mesh = model.mesh

        # Define the variational (projection problem)
        W = element_cls.function_space(mesh)

        u_soln, p_soln = model.u_soln, model.p_soln
        ff = model.generate_facet_function(mesh)
        ds = dolfin.Measure("ds", subdomain_data=ff)

        U = element_cls.create_solution_variable(W)
        weak_bcs = model.generate_bcs(ds)

        # Forms Stokes
        A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
        element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

        # Particle advector
        hmin = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())

        # Transfer the computed velocity function and compute functionals
        uh, ph = element_cls.get_velocity(U), element_cls.get_pressure(U)
        erroru_l2[j] = dolfin.errornorm(u_soln, uh, "l2", degree_rise=1)
        erroru_h1[j] = dolfin.errornorm(u_soln, uh, "h1", degree_rise=1)
        erroru_div[j] = dolfin.assemble(div(uh) ** 2 * dx) ** 0.5
        if ph is not None:
            errorp_l2[j] = dolfin.errornorm(p_soln, ph, "l2", degree_rise=1)
            errorp_h1[j] = dolfin.errornorm(p_soln, ph, "h1", degree_rise=1)
        hsizes[j] = hmin

    hrates = np.log(hsizes[:-1] / hsizes[1:])
    ratesu_l2 = np.log(erroru_l2[:-1] / erroru_l2[1:]) / hrates
    ratesu_h1 = np.log(erroru_h1[:-1] / erroru_h1[1:]) / hrates

    dolfin.info("errors u l2: %s" % str(erroru_l2))
    dolfin.info("rates u l2: %s" % str(ratesu_l2))
    dolfin.info("rates u h1: %s" % str(ratesu_h1))
    assert np.all(np.abs(ratesu_l2 - ul2_rate) < 0.1)
    assert np.all(np.abs(ratesu_h1 - uh1_rate) < 0.1)

    if ph is not None:
        ratesp_l2 = np.log(errorp_l2[:-1] / errorp_l2[1:]) / hrates
        ratesp_h1 = np.log(errorp_h1[:-1] / errorp_h1[1:]) / hrates
        dolfin.info("rates p l2: %s" % str(ratesp_l2))
        dolfin.info("rates p h1: %s" % str(ratesp_h1))

        assert np.all(np.abs(ratesp_l2 - pl2_rate) < 0.11)
        if ph1_rate is not None:
            assert np.all(np.abs(ratesp_h1 - ph1_rate) < 0.11)

    if isinstance(element_class, geopart.stokes.incompressible.HDG):
        assert np.all(np.abs(erroru_div) < 1e-11)


class MMSCompressible(geopart.stokes.compressible.CompressibleStokesModel):
    def __init__(self, comm, nx, ny):
        self.mesh = dolfin.RectangleMesh.create(
            comm, [dolfin.Point(0.0, 0.0), dolfin.Point(1.0, 1.0)],
            [nx, ny], dolfin.CellType.Type.triangle, "left/right")

        rho = dolfin.Expression("exp(-x[0] - x[1])",
                                degree=4, domain=self.mesh)
        rhou_soln_code = ("exp(-x[0] - x[1])*exp(2*(x[0] + x[1]))",
                          "-exp(-x[0] - x[1])*exp(2*(x[0] + x[1]))")
        u_soln_code = ("exp(2*(x[0] + x[1]))",
                       "-exp(2*(x[0] + x[1]))")
        p_soln_code = "2.0 * exp(x[0]) * sin(x[1]) " \
                      "+ 1.5797803888225995912 / 3.0"

        rhou_soln = dolfin.Expression(
            rhou_soln_code, degree=4, domain=self.mesh)
        u_soln = dolfin.Expression(u_soln_code, degree=4, domain=self.mesh)
        p_soln = dolfin.Expression(p_soln_code, degree=4, domain=self.mesh)

        eta = 1

        def F_v(rhou, grad_rhou, p_local):
            grad_u = (grad_rhou * rho - outer(rhou, grad(rho))) / rho ** 2
            eye = Identity(2)
            tau = 2 * eta * (sym(grad_u) - 1.0 / 3.0 * tr(grad_u) * eye)
            return tau - p_local * Identity(2)

        self.F_v = F_v
        f = - div(F_v(rho*u_soln, grad(rho*u_soln), p_soln))

        self.rho = rho
        self.u_soln = u_soln
        self.p_soln = p_soln
        self.rhou_soln = rhou_soln

        super().__init__(eta=eta, f=f, rho=rho)

    @staticmethod
    def generate_facet_function(mesh):
        ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        dolfin.CompiledSubDomain(
            "near(x[0], 0.0) or near(x[1], 0.0)").mark(ff, 1)
        dolfin.CompiledSubDomain(
            "near(x[0], 1.0) or near(x[1], 1.0)").mark(ff, 2)
        return ff

    def generate_bcs(self, ds, is_conservative):
        n = dolfin.FacetNormal(self.mesh)
        rho, u_soln, p_soln = self.rho, self.u_soln, self.p_soln
        rhou_soln = self.rhou_soln
        gN = self.F_v(rho*u_soln, grad(rho*u_soln), p_soln)*n
        if is_conservative:
            return [dg.DGDirichletBC(ds(1), rhou_soln),
                    dg.DGNeumannBC(ds(2), gN)]
        else:
            return [dg.DGDirichletBC(ds(1), u_soln),
                    dg.DGNeumannBC(ds(2), gN)]


compressible_elements_rates = [
    (geopart.stokes.compressible.TaylorHoodConservative, 3, 2, 2, 1),
    (geopart.stokes.compressible.TaylorHood, 3, 2, 2, 1),
    (geopart.stokes.compressible.HDGConservative, 2, 1, 1, None),
    (geopart.stokes.compressible.HDG, 2, 1, 1, None),
    (geopart.stokes.compressible.HDG2Conservative, 3, 2, 2, 1),
    (geopart.stokes.compressible.HDG2, 3, 2, 2, 1),
]


@pytest.mark.parametrize("run_data", compressible_elements_rates)
def test_stokes_compressible_convergence(run_data):
    element_class, ul2_rate, uh1_rate, pl2_rate, ph1_rate = run_data

    n_eles = [8, 16, 32]
    erroru_l2 = np.array([0.0] * len(n_eles), dtype=np.double)
    erroru_h1 = np.array([0.0] * len(n_eles), dtype=np.double)
    erroru_div = np.array([0.0] * len(n_eles), dtype=np.double)
    errorp_l2 = np.array([0.0] * len(n_eles), dtype=np.double)
    errorp_h1 = np.array([0.0] * len(n_eles), dtype=np.double)
    hsizes = np.array([0.0] * len(n_eles), dtype=np.double)

    for j, n_ele in enumerate(n_eles):
        element_cls = element_class()
        model = MMSCompressible(dolfin.MPI.comm_world, n_ele, n_ele)
        mesh = model.mesh

        # Define the variational (projection problem)
        W = element_cls.function_space(mesh)

        rhou_soln, p_soln = model.rhou_soln, model.p_soln
        ff = model.generate_facet_function(mesh)
        ds = dolfin.Measure("ds", subdomain_data=ff)

        U = element_cls.create_solution_variable(W)

        is_conservative = isinstance(
            element_cls, geopart.stokes.compressible.ConservativeFormulation)
        weak_bcs = model.generate_bcs(ds, is_conservative)

        # Forms Stokes
        A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
        element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

        # Particle advector
        hmin = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())

        # Transfer the computed velocity function and compute functionals
        if is_conservative:
            rhouh = element_cls.get_flux_soln(U)
        else:
            rhouh = model.rho * element_cls.get_flux_soln(U)
        ph = element_cls.get_pressure(U)

        # erroru_l2[j] = dolfin.errornorm(rhou_soln, rhouh, "l2", degree_rise=1)
        # erroru_h1[j] = dolfin.errornorm(rhou_soln, rhouh, "h1", degree_rise=1)
        erroru_l2[j] = dolfin.assemble((rhou_soln - rhouh)**2 * dx)**0.5
        erroru_h1[j] = dolfin.assemble(grad(rhou_soln - rhouh)**2 * dx)**0.5
        erroru_div[j] = dolfin.assemble(div(rhouh) ** 2 * dx) ** 0.5
        if ph is not None:
            errorp_l2[j] = dolfin.errornorm(p_soln, ph, "l2", degree_rise=1)
            errorp_h1[j] = dolfin.errornorm(p_soln, ph, "h1", degree_rise=1)
        hsizes[j] = hmin

    hrates = np.log(hsizes[:-1] / hsizes[1:])
    ratesu_l2 = np.log(erroru_l2[:-1] / erroru_l2[1:]) / hrates
    ratesu_h1 = np.log(erroru_h1[:-1] / erroru_h1[1:]) / hrates

    dolfin.info("errors u l2: %s" % str(erroru_l2))
    dolfin.info("rates u l2: %s" % str(ratesu_l2))
    dolfin.info("rates u h1: %s" % str(ratesu_h1))
    assert np.all(np.abs(ratesu_l2 - ul2_rate) < 0.1)
    assert np.all(np.abs(ratesu_h1 - uh1_rate) < 0.1)

    if ph is not None:
        ratesp_l2 = np.log(errorp_l2[:-1] / errorp_l2[1:]) / hrates
        ratesp_h1 = np.log(errorp_h1[:-1] / errorp_h1[1:]) / hrates
        dolfin.info("rates p l2: %s" % str(ratesp_l2))
        dolfin.info("rates p h1: %s" % str(ratesp_h1))

        assert np.all(ratesp_l2 > pl2_rate - 0.2)
        if ph1_rate is not None:
            assert np.all(ratesp_h1 > ph1_rate - 0.2)

    if isinstance(element_class, geopart.stokes.incompressible.HDG):
        assert np.all(np.abs(erroru_div) < 1e-11)
