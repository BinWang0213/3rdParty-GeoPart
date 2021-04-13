import matplotlib.pyplot as plt

import dolfin_dg.hdg_form
import numpy as np
from dolfin import *

import geopart.stokes.compressible


def run_experiment(element_class):

    n_eles = [4, 8, 16, 32]
    errors_rhou_l2 = np.zeros_like(n_eles, dtype=np.double)
    errors_rhou_h1 = np.zeros_like(n_eles, dtype=np.double)
    errors_u_l2 = np.zeros_like(n_eles, dtype=np.double)
    errors_u_h1 = np.zeros_like(n_eles, dtype=np.double)
    errors_div_u = np.zeros_like(n_eles, dtype=np.double)
    errors_div_rhou = np.zeros_like(n_eles, dtype=np.double)
    errors_div_u_exp = np.zeros_like(n_eles, dtype=np.double)
    errors_p_l2 = np.zeros_like(n_eles, dtype=np.double)
    errors_p_h1 = np.zeros_like(n_eles, dtype=np.double)
    hs = np.zeros_like(n_eles, dtype=np.double)


    for run_no, n_ele in enumerate(n_eles):

        mesh = UnitSquareMesh(n_ele, n_ele)
        # mesh = UnitDiscMesh.create(
        #     MPI.comm_self, int(n_ele), 1, 2)
        element_cls = element_class()

        t = Constant(1.0)
        x = SpatialCoordinate(mesh)
        # rho = Expression("exp(-x[0] - x[1])", degree=k+1, domain=mesh)
        # rhou_soln = Expression(("exp(-x[0] - x[1])*exp(2*(x[0] + x[1]))",
        #                         "-exp(-x[0] - x[1])*exp(2*(x[0] + x[1]))"),
        #                        degree=k+1, domain=mesh)
        # rho = Constant(1.0, cell=mesh.ufl_cell())
        # rhou_soln = Expression(("exp(2*(x[0] + x[1]))",
        #                         "-exp(2*(x[0] + x[1]))"),
        #                        degree=k+1, domain=mesh)

        # u_soln = Expression(("exp(2*(x[0] + x[1]))",
        #                      "-exp(2*(x[0] + x[1]))"),
        #                     degree=k + 2,
        #                     domain=mesh)
        # p_soln = Expression("2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
        #                     degree=k + 2, domain=mesh)

        # rho = Constant(1.0, cell=mesh.ufl_cell())
        # u_soln = as_vector((
        #     -x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5),
        #     x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)
        # ))
        # rhou_soln = rho*u_soln
        # rhou_soln_expr = Expression(
        #     ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
        #      "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
        #     t=1.0, degree=6, domain=mesh)
        # u_soln_expr = Expression(
        #     ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
        #      "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
        #     t=1.0, degree=6, domain=mesh)

        # rho = x[0] * x[0] + x[1] * x[1] + 1.0
        # u_soln = as_vector((
        #     -x[1] * sqrt(x[0] * x[0] + x[1] * x[1]) * (pow(cos(t), 2) + 0.5),
        #     x[0] * sqrt(x[0] * x[0] + x[1] * x[1]) * (pow(cos(t), 2) + 0.5)
        # ))
        # rhou_soln = rho * u_soln
        # rhou_soln_expr = Expression(
        #     ("-((x[0]*x[0] + x[1]*x[1]) + 1.0)*x[1]*sqrt(x[0]*x[0] + x["
        #      "1]*x[1])*(pow(cos(t), 2) + 0.5)",
        #      "((x[0]*x[0] + x[1]*x[1]) + 1.0)*x[0]*sqrt(x[0]*x[0] + x["
        #      "1]*x[1])*(pow(cos(t), 2) + 0.5)"),
        #     t=1.0, degree=6, domain=mesh)
        # u_soln_expr = Expression(
        #     ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
        #      "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
        #     t=1.0, degree=6, domain=mesh)

        c = Constant(1e5)
        rho = sqrt(x[0]*x[0] + x[1]*x[1]) + c
        rho_expr = Expression("sqrt(x[0]*x[0] + x[1]*x[1]) + c", degree=6, c=c)
        u_soln = as_vector((
            -x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5),
            x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)
        ))
        rhou_soln = rho*u_soln
        rhou_soln_expr = Expression(
            ("-(sqrt(x[0]*x[0] + x[1]*x[1]) + c)*x[1]*sqrt(x[0]*x[0] + x["
             "1]*x[1])*(pow(cos(t), 2) + 0.5)",
             "(sqrt(x[0]*x[0] + x[1]*x[1]) + c)*x[0]*sqrt(x[0]*x[0] + x["
             "1]*x[1])*(pow(cos(t), 2) + 0.5)"),
            t=1.0, degree=6, domain=mesh, c=c)
        u_soln_expr = Expression(
            ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
             "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
            t=1.0, degree=6, domain=mesh)


        p_soln = Constant(0.0)

        ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        CompiledSubDomain("near(x[0], 0.0) or near(x[1], 0.0)").mark(ff, 1)
        CompiledSubDomain("near(x[0], 1.0) or near(x[1], 1.0)").mark(ff, 2)
        # CompiledSubDomain("on_boundary").mark(ff, 1)
        # CompiledSubDomain("on_boundary and (x[0] < 0.0)").mark(ff, 2)
        ds = Measure("ds", subdomain_data=ff)
        dsD, dsN = ds(1), ds(2)

        dx = Measure("dx", domain=mesh)

        W = element_cls.function_space(mesh)
        U = element_cls.create_solution_variable(W)

        # -- Automatic
        def F_v(u, grad_u, p_local):
            return 2*(sym(grad_u) - 1.0/3.0*tr(grad_u)*Identity(2)) - p_local*Identity(2)

        is_conservative = isinstance(
            element_cls, geopart.stokes.compressible.ConservativeFormulation)
        n = FacetNormal(mesh)
        gN = F_v(u_soln, grad(u_soln), p_soln)*n
        if is_conservative:
            weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), rhou_soln_expr),
                        dolfin_dg.DGNeumannBC(ds(2), gN)]
        else:
            weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), u_soln_expr),
                        dolfin_dg.DGNeumannBC(ds(2), gN)]
        f = - div(F_v(u_soln, grad(u_soln), p_soln))
        model = geopart.stokes.compressible.CompressibleStokesModel(
            eta=Constant(1.0), f=f, rho=rho)

        A, b = PETScMatrix(), PETScVector()
        element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)
        # element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

        if is_conservative:
            rhou = element_cls.get_flux_soln(U)
            u = element_cls.get_velocity(U, model=model)
        else:
            rhou = model.rho * element_cls.get_flux_soln(U)
            u = element_cls.get_flux_soln(U)
        p = element_cls.get_pressure(U)

        class VelFromRhouFunction(UserExpression):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def eval_cell(self, values, x, cell):
                rhou.eval_cell(values, x, cell)
                values /= (x[0]*x[0] + x[1]*x[1])**0.5 + 1.0

            def value_shape(self):
                return (2,)

        # hi_ele = VectorElement("DG", mesh.ufl_cell(), 4)
        # u_hi = VelFromRhouFunction(
        #     element=hi_ele,
        #     domain=mesh)
        # u = u_hi
        # u_hi_expr = Expression(("rhou[0]/rho", "rhou[1]/rho"), element=hi_ele,
        #                        rhou=rhou, rho=rho_expr)
        # u = interpolate(u_hi, FunctionSpace(mesh, hi_ele))
        # u = project(u, element_cls.velocity_function_space(mesh))
        # u = project(u, FunctionSpace(mesh, hi_ele))

        errors_rhou_l2[run_no] = assemble(
            dot(rhou - rhou_soln, rhou - rhou_soln) * dx) ** 0.5
        errors_rhou_h1[run_no] = assemble(
            inner(grad(rhou - rhou_soln), grad(rhou - rhou_soln)) * dx) ** 0.5

        errors_u_l2[run_no] = assemble(
            dot(u - u_soln, u - u_soln) * dx) ** 0.5
        errors_u_h1[run_no] = assemble(
            inner(grad(u - u_soln), grad(u - u_soln)) * dx) ** 0.5

        errors_div_u[run_no] = assemble((div(u) - div(u_soln))**2*dx)**0.5
        errors_div_rhou[run_no] = assemble(div(rhou)**2*dx)**0.5
        errors_div_u_exp[run_no] = assemble(
            (div(u) - dot(u, grad(rho))/rho)**2*dx)**0.5

        errors_p_l2[run_no] = assemble((p - p_soln) ** 2 * dx) ** 0.5
        errors_p_h1[run_no] = assemble(
            inner(grad(p - p_soln), grad(p - p_soln))*dx) ** 0.5

        hs[run_no] = MPI.min(mesh.mpi_comm(), mesh.hmin())

    return hs, errors_rhou_l2, errors_rhou_h1, errors_u_l2, errors_u_h1, \
           errors_p_l2, errors_p_h1, errors_div_u, errors_div_rhou, \
           errors_div_u_exp


def compute_convergence_rates(e, h):
    hrate = np.log(h[:-1] / h[1:])
    rates = np.log(e[:-1] / e[1:]) / hrate
    return rates

if __name__ == "__main__":


    element_classes = (
        geopart.stokes.compressible.TaylorHoodConservative,
        # geopart.stokes.compressible.TaylorHood,
        geopart.stokes.compressible.HDG2Conservative,
        # geopart.stokes.compressible.HDG2,
        # geopart.stokes.compressible.HDGDivu2,
    )

    for element_class in element_classes:
        error_data = run_experiment(element_class)
        (
            hs,
            errors_rhou_l2, errors_rhou_h1,
            errors_u_l2, errors_u_h1,
            errors_p_l2, errors_p_h1,
            errors_div_u, errors_div_rhou, errors_div_u_exp
        ) = error_data

        print(element_class.__name__)
        print("rhou L2:", compute_convergence_rates(errors_rhou_l2, hs))
        print("rhou H1:", compute_convergence_rates(errors_rhou_h1, hs))

        print("u L2:", compute_convergence_rates(errors_u_l2, hs))
        print("u H1:", compute_convergence_rates(errors_u_h1, hs))

        print("div(rhou) L2:", compute_convergence_rates(errors_div_rhou, hs))
        print("div(u) L2:", compute_convergence_rates(errors_div_u, hs))
        print(list(map(lambda x: "%.3e" % x, errors_div_u)))
        print("div(u) - u . grad(rho) / rho L2:",
              compute_convergence_rates(errors_div_u_exp, hs))

        print("p L2:", compute_convergence_rates(errors_p_l2, hs))
        print("p H2:", compute_convergence_rates(errors_p_h1, hs))

        plt.figure(1)
        plt.loglog(hs, errors_u_l2, "-", label=r"$e(u)$")

        plt.figure(2)
        plt.loglog(hs, errors_rhou_h1, "-", label=r"$e_{H^1}(\rho u)$")

        plt.figure(3)
        plt.loglog(hs, errors_div_u, "-", label=r"$e(\nabla \cdot u)$")
        # plt.loglog(hs, errors_div_rhou, "-", label=r"$e(\nabla \cdot \rho u)$")
        # plt.loglog(hs, errors_div_u_exp,
        #            "-", label=r"$e(\nabla \cdot u "
        #                       r"- \frac{u \cdot \nabla \rho}{\rho})$")

    plt.figure(1)
    plt.xlabel("$h$")
    plt.ylabel(r"$\Vert \mathbf{u} - \mathbf{u}_h \Vert_{L_2(\Omega)}$")
    plt.legend(("TH", "HDG2"))
    plt.grid()
    plt.savefig("u_l2.png", bbox_inches="tight")

    plt.figure(2)
    plt.xlabel("$h$")
    plt.ylabel(r"$\Vert \nabla \mathbf{u} - \nabla \mathbf{u}_h \Vert_{L_2("
               r"\Omega)}$")
    plt.legend(("TH", "HDG2"))
    plt.grid()
    plt.savefig("u_h1.png", bbox_inches="tight")

    plt.figure(3)
    plt.xlabel("$h$")
    plt.ylabel(r"$\Vert \nabla \cdot \mathbf{u} - \nabla \cdot \mathbf{u}_h "
               r"\Vert_{L_2("
               r"\Omega)}$")
    plt.legend(("TH", "HDG2"))
    plt.grid()
    plt.savefig("divu_l2.png", bbox_inches="tight")

    plt.show()
