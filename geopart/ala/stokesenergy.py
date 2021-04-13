import abc
import typing

import numpy

import dolfin
import dolfin_dg
import dolfin_dg.operators
import dolfin_dg.hdg_form
import dolfin_dg.dolfin.hdg_newton
import ufl
from dolfin import dx
from ufl import grad, dot

import leopart

import geopart.particleprojection
import geopart.elem
import geopart.composition.incompressible
import geopart.timings


class StokesHeatPDEConstrainedHDG1(
        geopart.particleprojection.AbstractParticleProjectionScheme):

    def degree(self):
        return 1

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        self.dt_ufl = dt_ufl
        self.u_vec = u_vec
        self.property_idx = property_idx
        self.pde_projection = None
        self.zeta_val = zeta_val
        self.ptcls = ptcls

        self.theta = dolfin.Constant(1.0)  # Backward Euler & 2nd order method
        self.theta_L = dolfin.Constant(1.0)  # TODO: Must change to 0.5 on 2nd
        # step

        self.periodic = None  # TODO: Account for BCs

        mesh = ptcls.mesh()
        T_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
        We, Wbare, Qe, Qbare, Se, Sbare = self.ufl_element(mesh)

        # -- Advection spaces
        self.V_adv = dolfin.FunctionSpace(mesh, Se)
        self.langrange_space = dolfin.FunctionSpace(mesh, T_e)
        Vbar_adv = dolfin.FunctionSpace(mesh, Sbare) if self.periodic is None \
            else dolfin.FunctionSpace(mesh, Sbare,
                                      constrained_domain=self.periodic)
        self.Vbar_adv = Vbar_adv
        FuncSpace_adv = {'FuncSpace_local': self.V_adv,
                         'FuncSpace_lambda': self.langrange_space,
                         'FuncSpace_bar': self.Vbar_adv}
        self.forms_pde_map = \
            geopart.composition.incompressible.FormsPDEMapExtended(
                mesh, FuncSpace_adv)

        # -- Diffusion spaces
        self.V = dolfin.FunctionSpace(mesh, ufl.MixedElement([We, Qe, Se]))
        self.Vbar = dolfin.FunctionSpace(mesh, ufl.MixedElement([
            Wbare, Qbare, Sbare]))

        # This is messy
        self.T0_a = dolfin.Function(self.V_adv)
        self.T0_abar = dolfin.Function(self.Vbar_adv)
        self.dTh0 = dolfin.Function(self.V_adv)
        self.dTh00 = dolfin.Function(self.V_adv)

        self.prop_idxs = \
            numpy.array([self.property_idx, self.property_idx + 1],
                        dtype=numpy.uintp)

        self.solver = None

    def ufl_element(self, mesh: dolfin.Mesh):
        k = self.degree()

        # Momentum spaces
        We = ufl.VectorElement("DG", mesh.ufl_cell(), k)
        Wbare = ufl.VectorElement("CG", mesh.ufl_cell(), k)["facet"]

        # Pressure spaces
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), k - 1)
        Qbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), k)

        # Temperature spaces
        Se = ufl.FiniteElement("DG", mesh.ufl_cell(), k)
        # Sbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), k)
        Sbare = ufl.FiniteElement("CG", mesh.ufl_cell(), k)["facet"]
        return We, Wbare, Qe, Qbare, Se, Sbare

    def function_space(self):
        return self.V, self.Vbar

    def temperature_advection_space(self):
        return self.V_adv, self.Vbar_adv

    def velocity_function_space(self, mesh):
        return dolfin.FunctionSpace(mesh, self.ufl_element(mesh)[0])

    def velocity_sub_space(self, W: dolfin.FunctionSpace):
        return W[0].sub(0)

    def temperature_sub_space(self, W: dolfin.FunctionSpace):
        return W[0].sub(2)

    def create_solution_variable(self, W: dolfin.FunctionSpace):
        return dolfin.Function(W[0]), dolfin.Function(W[1])

    def get_velocity(self, U: dolfin.Function):
        return dolfin.split(U[0])[0]

    def get_velocity_sub_function(self, U: dolfin.Function):
        return U[0].sub(0)

    def get_pressure(self, U: dolfin.Function):
        return dolfin.split(U[0])[1]

    def get_pressure_sub_function(self, U: dolfin.Function):
        return U[0].sub(0)

    def get_temperature(self, U: dolfin.Function):
        return dolfin.split(U[0])[2]

    def get_temperature_sub_function(self, U: dolfin.Function):
        return U[0].sub(2)

    def compute_cfl_dt(self, u_vec, hmin, c_cfl):
        max_u_vec = u_vec.vector().norm("linf")
        return c_cfl * hmin / max_u_vec

    @geopart.timings.apply_dolfin_timer
    def generate_diffusion_forms(
            self, W: dolfin.FunctionSpace, U: dolfin.Function,
            Tstar: dolfin.Function,
            stokes_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            heat_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            model):
        k = self.degree()

        W_, Wbar_ = W
        U_, Ubar_ = U

        V_, Vbar_ = dolfin.TestFunction(W_), dolfin.TestFunction(Wbar_)
        v, q, s = ufl.split(V_)
        vbar, qbar, sbar = ufl.split(Vbar_)

        Tstar, Tstarbar = Tstar

        alpha_u = dolfin.Constant(12 * k ** 2)
        rho = model.rho

        rhou, p, T = ufl.split(U[0])
        rhoubar, pbar, Tbar = ufl.split(U[1])

        eta = model.eta
        f = model.f_u
        f_T = model.f_T
        kappa = model.kappa

        dx, dS, ds = ufl.dx, ufl.dS, ufl.ds

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        def F_v(rhou, grad_rhou, p_local=None):
            if p_local is None:
                p_local = pbar
            grad_u = (grad_rhou*rho - ufl.outer(rhou, ufl.grad(rho)))/rho**2
            eye = ufl.Identity(2)
            tau = 2*eta*(ufl.sym(grad_u) - 1.0/3.0*ufl.tr(grad_u) * eye)
            sigma = tau - p_local * eye
            return sigma

        mesh = W_.mesh()
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        penalty_u = alpha_u / h
        G = dolfin_dg.homogeneity_tensor(F_v, rhou)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, rhou, rhoubar, v, vbar, penalty_u, G, n)

        F = ufl.inner(F_v(rhou, ufl.grad(rhou), p), ufl.grad(v)) * dx
        F += hdg_term.face_residual(dS, ds)
        F += - ufl.inner(f, v) * dx

        # Continuity
        F += ufl.inner(q, ufl.div(rhou)) * dx
        F += facet_integral(ufl.inner(ufl.dot(rhou - rhoubar, n), qbar))

        # Neumann BCs
        for bc in stokes_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= ufl.dot(bc.get_function(), vbar) * bc.get_boundary()

        # Heat
        def F_v_T(u, grad_u):
            return kappa * grad_u

        alpha_T = dolfin.Constant(24 * k ** 2)
        penalty_T = alpha_T / h
        G_T = dolfin_dg.homogeneity_tensor(F_v_T, T)
        hdg_term_T = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v_T, T, Tbar, s, sbar, penalty_T, G_T, n)

        dTdt = (T - Tstar) / self.dt_ufl

        F += rho * dTdt * s * dx
        F += ufl.inner(F_v_T(T, grad(T)), grad(s)) * dx
        F += hdg_term_T.face_residual(dS, ds)
        F += - ufl.inner(f_T, s) * dx

        # Construct local and global block components
        Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
        J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

        strong_bcs = []
        for (bcs, fspace) in [(stokes_bcs, Wbar_.sub(0)),
                              (heat_bcs, Wbar_.sub(2))]:
            strong_bcs += [dolfin.DirichletBC(fspace,
                                             bc.get_function(),
                                             bc.get_boundary().subdomain_data(),
                                             bc.get_boundary().subdomain_id())
                          for bc in bcs
                          if isinstance(bc, dolfin_dg.DGDirichletBC)]
            strong_bcs += [dolfin.DirichletBC(fspace.sub(bc.component),
                                              dolfin.Constant(0.0),
                                              bc.get_boundary().subdomain_data(),
                                              bc.get_boundary().subdomain_id())
                           for bc in bcs if
                           isinstance(bc, dolfin_dg.DGDirichletNormalBC)]
        return (Fr, J), strong_bcs

    @geopart.timings.apply_dolfin_timer
    def solve_diffusion(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                        Tstar: dolfin.Function,
                        mats: typing.Sequence[
                            typing.Union[dolfin.Vector, dolfin.Matrix]],
                        stokes_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                        heat_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                        model, **solver_options) -> None:
        if self.solver is None:
            (Fr, J), strong_bcs = self.generate_diffusion_forms(
                W, U, Tstar, stokes_bcs, heat_bcs, model)
            solver = \
                dolfin_dg.dolfin.hdg_newton.StaticCondensationNewtonSolver(
                    Fr, J, strong_bcs, **solver_options)
            self.solver = solver
        U_, Ubar_ = U
        self.solver.solve(U_, Ubar_)


    @geopart.timings.apply_dolfin_timer
    def project_advection(self, Ustar: dolfin.Function,
                          heat_bcs) -> None:
        Tstar, Tstarbar = Ustar
        if self.pde_projection is None:
            forms_pde_map = self.forms_pde_map

            psi_star = self.T0_a + self.dt_ufl * (
                    (1 - self.theta_L) * self.dTh00 + self.theta_L * self.dTh0)
            forms_pde = forms_pde_map.forms_theta_linear(
                self.T0_a, self.u_vec, self.dt_ufl, theta_map=self.theta,
                theta_L=self.theta_L, dpsi0=self.dTh0, dpsi00=self.dTh00,
                zeta=dolfin.Constant(self.zeta_val), psi_star=psi_star)

            strong_bcs = [dolfin.DirichletBC(
                Ustar[1].function_space(), bc.get_function(),
                bc.get_boundary().subdomain_data(),
                bc.get_boundary().subdomain_id())
                for bc in heat_bcs
                if isinstance(bc, dolfin_dg.DGDirichletBC)]

            strong_bcs += [bc for bc in heat_bcs
                           if isinstance(bc, dolfin.DirichletBC)]

            pde_projection = leopart.PDEStaticCondensation(
                self.ptcls.mesh(), self.ptcls,
                forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                forms_pde['H_a'],
                forms_pde['B_a'],
                forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                strong_bcs, self.property_idx)
            self.pde_projection = pde_projection

        self.pde_projection.assemble(True, True)
        self.pde_projection.solve_problem(Tstarbar.cpp_object(),
                                          Tstar.cpp_object(),
                                          "mumps", "default")

    @geopart.timings.apply_dolfin_timer
    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        T = U[0]
        Tstar = Ustar[0]

        self.T0_a.vector()[:] = Tstar.vector()
        self.dTh00.vector()[:] = self.dTh0.vector()
        self.dTh0.vector()[:] = \
            (T.vector() - Tstar.vector())/dt

        self.ptcls.increment(self.dTh0, self.prop_idxs, theta_p, step, dt)

    def get_heat(self, U: dolfin.Function):
        return U[0]

class StokesHeatPDEConstrainedHDG2(StokesHeatPDEConstrainedHDG1):

    def degree(self):
        return 2


class StokesHeatLeastSquaresDG1(StokesHeatPDEConstrainedHDG1):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        super().__init__(ptcls, u_vec, dt_ufl, property_idx, zeta_val)
        Vh = self.temperature_advection_space()[0]
        self.l2p = leopart.l2projection(ptcls, Vh, property_idx)

    def degree(self):
        return 1

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, Ustar: dolfin.Function,
                          bcs) -> None:
        Tstar, Tstarbar = Ustar
        self.l2p.project(Tstar.cpp_object())


class StokesHeatLeastSquaresDG2(StokesHeatLeastSquaresDG1):

    def degree(self):
        return 2
