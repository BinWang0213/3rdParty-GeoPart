import matplotlib.pyplot as plt

import ufl
import dolfin
import dolfin_dg

import geopart.stokes.compressible


def run_experiment(element_cls):
    mesh = dolfin.UnitSquareMesh(32, 32, "left/right")

    eta = dolfin.Constant(1.0)
    f = dolfin.Constant((0.0, 0.0))
    rho = dolfin.Expression("A*(1.0 - x[1]) + 1.0", A=1.0, degree=4,
                            domain=mesh)
    stokes_model = geopart.stokes.compressible.CompressibleStokesModel(
        eta=eta, f=f, rho=rho)

    W = element_cls.function_space(mesh)
    U = element_cls.create_solution_variable(W)

    ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    dolfin.CompiledSubDomain("near(x[0], 0.0) or near(x[0], 1.0)").mark(ff, 1)
    dolfin.CompiledSubDomain("near(x[1], 0.0)").mark(ff, 2)
    dolfin.CompiledSubDomain("near(x[1], 1.0)").mark(ff, 3)
    ds = ufl.Measure("ds", subdomain_data=ff)

    n = ufl.FacetNormal(mesh)
    soln_flux = element_cls.get_flux_soln(U)
    normal_bc_x = dolfin_dg.DGDirichletNormalBC(
        ds(1), dolfin_dg.tangential_proj(soln_flux, n))
    normal_bc_x.component = 0
    normal_bc_y = dolfin_dg.DGDirichletNormalBC(
        ds(2), dolfin_dg.tangential_proj(soln_flux, n))
    normal_bc_y.component = 1

    is_conservative = isinstance(
        element_cls, geopart.stokes.compressible.ConservativeFormulation)
    if is_conservative:
        rhou_0 = dolfin.project(rho*dolfin.Constant((1.0, 0.0)),
                                element_cls.velocity_function_space(mesh))
        lid_bc = dolfin_dg.DGDirichletBC(ds(3), rhou_0)
    else:
        u_0 = dolfin.project(dolfin.Constant((1.0, 0.0)),
                                element_cls.velocity_function_space(mesh))
        lid_bc = dolfin_dg.DGDirichletBC(ds(3), u_0)

    weak_bcs = [normal_bc_x, normal_bc_y, lid_bc]

    A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
    element_cls.solve_stokes(W, U, (A, b), weak_bcs, stokes_model)

    if is_conservative:
        rhou = element_cls.get_flux_soln(U)
        u = dolfin.project(rhou/rho, element_cls.velocity_function_space(mesh))
    else:
        u = element_cls.get_flux_soln(U)
        rhou = rho*u

    div_rhou_error = dolfin.assemble(ufl.div(rhou)**2*ufl.dx)**0.5
    div_u_error = dolfin.assemble(ufl.div(u)**2*ufl.dx)**0.5
    rhou_error = dolfin.assemble((rhou - rho*u)**2*ufl.dx)**0.5

    print(f"{element_cls.__class__.__name__:<25}: ",
          f"‖∇⋅(ρu)‖₂={div_rhou_error:<.3e} ",
          f"‖∇⋅(ρ * Π(ρu/ρ))‖₂={div_u_error:<.3e} ",
          f"‖ (ρu)ₕ - ρ⋅uₕ ‖₂={rhou_error:<.3e}")

    return div_rhou_error


if __name__ == "__main__":
    element_classes = (
        geopart.stokes.compressible.TaylorHood(),
        geopart.stokes.compressible.HDG(),
        geopart.stokes.compressible.HDG2(),
        geopart.stokes.compressible.TaylorHoodConservative(),
        geopart.stokes.compressible.HDGConservative(),
        geopart.stokes.compressible.HDG2Conservative()
    )

    div_rhou_errors = [None] * len(element_classes)

    for j, element_class in enumerate(element_classes):
        div_rhou_errors[j] = run_experiment(element_class)

    x_ticks = list(range(len(div_rhou_errors)))
    plt.bar(x_ticks, div_rhou_errors)
    plt.yscale("log")
    plt.xticks(
        x_ticks, list(map(lambda cls: type(cls).__name__, element_classes)))
    plt.ylabel(r"$\Vert \nabla \cdot (\rho u)_h \Vert_{L_2}$")
    plt.xlabel("Stokes numerical scheme")
    plt.show()