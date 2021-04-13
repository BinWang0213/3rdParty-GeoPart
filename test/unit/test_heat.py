import dolfin
import dolfin_dg
import leopart
import numpy as np
import pytest
import ufl

import geopart.energy.heat
import geopart.energy.poisson

element_data = [
    (geopart.energy.poisson.PoissonCG1, 2, 1),
    (geopart.energy.poisson.PoissonCG2, 3, 2),
]


@pytest.mark.parametrize("element_datum", element_data)
def test_steady_temperature_formulation(element_datum):
    element_class, l2rate, h1rate = element_datum

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

    assert np.all(l2rates > l2rate - 0.2)
    assert np.all(h1rates > h1rate - 0.2)


def generate_T_d(mesh, t_ufl, kappa):
    # Needs to be very smooth to make the test cheap with coarse meshes
    sigma = dolfin.Constant(0.3)
    x0, y0 = (0.5, 0.5)
    a = "2*pow(sigma, 2)"
    b = "4*kappa"
    T_exact = f"{a} / ({a} + {b} * t) " \
              f"* exp(-(pow(x[0] - x0, 2) + pow(x[1] - y0, 2))" \
              f"/({a} + {b} * t))"

    T_soln_bc = dolfin.Expression(
        T_exact, degree=6, sigma=sigma,
        x0=x0, y0=y0, kappa=kappa, domain=mesh, t=t_ufl)

    return T_soln_bc


heat_on_particles_data = [
    (geopart.energy.heat.HeatLeastSquaresHDG1, 2, 1),
    (geopart.energy.heat.HeatLeastSquaresHDG2, 3, 2),
    (geopart.energy.heat.HeatPDEConstrainedHDG1, 2, 1),
    (geopart.energy.heat.HeatPDEConstrainedHDG2, 3, 2)
]


@pytest.mark.parametrize("element_datum", heat_on_particles_data)
def test_transient_heat_stationary_particles(element_datum):
    element_class, l2rate, h1rate = element_datum

    n_eles = np.array((4, 8), dtype=np.int)
    u_errors = np.zeros_like(n_eles, dtype=np.double)
    u_h1errors = np.zeros_like(n_eles, dtype=np.double)
    h = np.zeros_like(n_eles, dtype=np.double)
    dt_fixed_vals = list(0.08 * 2 ** -a for a in range(len(n_eles)))

    dt_ufl = dolfin.Constant(0.0)
    t_ufl = dolfin.Constant(0.0)

    for i, n_ele in enumerate(n_eles):
        dt_ufl.assign(0.0)
        t_ufl.assign(0.0)

        mesh = dolfin.UnitSquareMesh(n_ele, n_ele, "left/right")
        h[i] = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())

        kappa = dolfin.Constant(1e-3)
        Q = dolfin.Constant(0.0)
        heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)

        xp = np.array([[0.0, 0.0]], dtype=np.float_)
        xs = [np.zeros((len(xp), 1), dtype=np.float_),
              np.zeros((len(xp), 1), dtype=np.float_)]
        ptcls = leopart.particles(xp, xs, mesh)

        heat_cls = element_class(ptcls, dolfin.Constant((0, 0)), dt_ufl, 1)
        Wh = heat_cls.function_space()
        Tstar, Tstarbar = heat_cls.create_solution_variable(Wh)
        T, Tbar = heat_cls.create_solution_variable(Wh)
        T_D_bc = generate_T_d(mesh, t_ufl, kappa)

        heat_cls.T0_a.assign(T_D_bc)

        npart = 25 * max(Wh[0].ufl_element().degree(), 1)
        leopart.AddDelete(ptcls, npart, npart, [T]).do_sweep()

        T_exact = dolfin.Function(Wh[0])
        dolfin.project(T_D_bc, T_exact.function_space(), function=T_exact)
        ptcls.interpolate(T_exact, 1)

        bcs_d = [dolfin.DirichletBC(Wh[1], T_D_bc, "on_boundary")]
        A_heat, b_heat = dolfin.PETScMatrix(), dolfin.PETScVector()

        t = 0.0
        last_step = False
        t_max = 2.0
        for j in range(500):
            step = j+1
            dt = dt_fixed_vals[i]

            if t + dt > t_max - 1e-9:
                dt = t_max - t
                last_step = True

            dt_ufl.assign(dt)
            t += dt
            t_ufl.assign(t)

            heat_cls.project_advection([Tstar, Tstarbar], [])
            heat_cls.solve_diffusion(
                Wh, [T, Tbar], [Tstar, Tstarbar], [A_heat, b_heat], bcs_d,
                heat_model)

            theta_p = 0.5
            heat_cls.update_field_and_increment_particles(
                [T, Tbar], [Tstar, Tstarbar], theta_p, step, dt)

            if step == 2:
                heat_cls.theta_L.assign(0.5)

            if last_step:
                break

        u_errors[i] = dolfin.assemble(
            (T_D_bc - T)**2*dolfin.dx)**0.5
        u_h1errors[i] = dolfin.assemble(
            ufl.grad(T_D_bc - T)**2*dolfin.dx)**0.5

    l2rates = np.log(u_errors[1:]/u_errors[:-1])/np.log(h[1:]/h[:-1])
    h1rates = np.log(u_h1errors[1:]/u_h1errors[:-1])/np.log(h[1:]/h[:-1])

    assert np.all(l2rates > l2rate - 0.2)
    assert np.all(h1rates > h1rate - 0.2)
