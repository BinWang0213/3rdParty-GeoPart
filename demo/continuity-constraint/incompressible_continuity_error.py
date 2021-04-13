import matplotlib.pyplot as plt

import ufl
import dolfin
import dolfin_dg

import geopart.stokes
import geopart.stokes.incompressible


def run_experiment(element_cls, stokes_model):
    mesh = dolfin.UnitSquareMesh(32, 32, "left/right")

    W = element_cls.function_space(mesh)
    U = element_cls.create_solution_variable(W)

    ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    dolfin.CompiledSubDomain("near(x[0], 0.0) or near(x[0], 1.0)").mark(ff, 1)
    dolfin.CompiledSubDomain("near(x[1], 0.0)").mark(ff, 2)
    dolfin.CompiledSubDomain("near(x[1], 1.0)").mark(ff, 3)
    ds = ufl.Measure("ds", subdomain_data=ff)

    n = ufl.FacetNormal(mesh)
    u = element_cls.get_velocity(U)
    normal_bc_x = dolfin_dg.DGDirichletNormalBC(
        ds(1), dolfin_dg.tangential_proj(u, n))
    normal_bc_x.component = 0
    normal_bc_y = dolfin_dg.DGDirichletNormalBC(
        ds(2), dolfin_dg.tangential_proj(u, n))
    normal_bc_y.component = 1
    lid_bc = dolfin_dg.DGDirichletBC(ds(3), dolfin.Constant((1.0, 0.0)))
    weak_bcs = [normal_bc_x, normal_bc_y, lid_bc]

    A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
    element_cls.solve_stokes(W, U, (A, b), weak_bcs, stokes_model)

    u = element_cls.get_velocity(U)
    div_u_error = dolfin.assemble(ufl.div(u)**2*ufl.dx)**0.5
    print(f"{element_cls.__class__.__name__:<10}: ",
          f"‖∇⋅(u)‖₂={div_u_error:<.3e}")

    return div_u_error


if __name__ == "__main__":
    eta = dolfin.Constant(1.0)
    f = dolfin.Constant((0.0, 10.0))
    stokes_model = geopart.stokes.StokesModel(eta, f)

    element_classes = (
        geopart.stokes.incompressible.TaylorHood(),
        geopart.stokes.incompressible.HDG(),
        geopart.stokes.incompressible.HDG2(),
        geopart.stokes.incompressible.Mini(),
        geopart.stokes.incompressible.DG(),
        geopart.stokes.incompressible.P2BDG1(),
        geopart.stokes.incompressible.P2DG0(),
    )

    div_u_errors = [None]*len(element_classes)

    for j, element_class in enumerate(element_classes):
        div_u_errors[j] = run_experiment(element_class, stokes_model)

    x_ticks = list(range(len(div_u_errors)))
    plt.bar(x_ticks, div_u_errors)
    plt.yscale("log")
    plt.xticks(
        x_ticks, list(map(lambda cls: type(cls).__name__, element_classes)))
    plt.ylabel(r"$\Vert \nabla \cdot u_h \Vert_{L_2}$")
    plt.xlabel("Stokes numerical scheme")
    plt.show()