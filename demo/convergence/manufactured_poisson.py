import os

import dolfin
import dolfin_dg
import numpy as np
import ufl

import geopart.energy.heat
import geopart.energy.poisson

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False


def run_experiment(element_class):
    n_eles = np.array((4, 8, 16), dtype=np.int)
    u_errors = np.zeros_like(n_eles, dtype=np.double)
    u_h1errors = np.zeros_like(n_eles, dtype=np.double)
    h = np.zeros_like(n_eles, dtype=np.double)
    for j, n_ele in enumerate(n_eles):
        mesh = dolfin.UnitSquareMesh(n_ele, n_ele, "left/right")
        h[j] = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())

        Q = dolfin.Expression("2*pi*pi*sin(pi*x[0])*sin(pi*x[1])",
                              degree=4, domain=mesh)
        heat_model = geopart.energy.heat.HeatModel(kappa=1, Q=Q)

        element_cls = element_class()
        W = element_cls.function_space(mesh)
        U = element_cls.create_solution_variable(W)

        ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        dolfin.CompiledSubDomain("on_boundary").mark(ff, 1)
        ds = ufl.Measure("ds", subdomain_data=ff)

        bc = dolfin_dg.DGDirichletBC(ds(1), dolfin.Constant(0.0))
        weak_bcs = [bc]

        A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
        element_cls.solve_heat(W, U, (A, b), weak_bcs, heat_model)

        u_soln = dolfin.Expression("sin(pi*x[0])*sin(pi*x[1])",
                                   degree=4, domain=mesh)
        u_errors[j] = dolfin.assemble((U - u_soln)**2*dolfin.dx)**0.5
        u_h1errors[j] = dolfin.assemble(ufl.grad(U - u_soln)**2*dolfin.dx)**0.5

    l2rates = np.log(u_errors[1:]/u_errors[:-1])/np.log(h[1:]/h[:-1])
    h1rates = np.log(u_h1errors[1:]/u_h1errors[:-1])/np.log(h[1:]/h[:-1])

    dolfin.info(f"Element class {element_cls.__class__.__name__}")
    dolfin.info(f"l2 error: {u_errors}, h1 error: {u_h1errors}")
    dolfin.info(f"l2 rates: {l2rates}, h1 rates {h1rates}")

    if IS_TEST_ENV:
        assert np.all(l2rates > element_cls.degree() + 1 - 0.1)
        assert np.all(h1rates > element_cls.degree() - 0.1)


if __name__ == "__main__":

    element_classes = (
        geopart.energy.poisson.PoissonCG1,
        geopart.energy.poisson.PoissonCG2
    )

    div_u_errors = [None]*len(element_classes)

    for element_class in element_classes:
        run_experiment(element_class)

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
