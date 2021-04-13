import os

import meshio
import pygmsh
import dolfin
import dolfin_dg
import leopart
import numpy as np
import ufl
from dolfin import MPI
from ufl import dx

import geopart.composition.compressible
import geopart.energy.heat
import geopart.projector
import geopart.stokes.compressible

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

use_fixed_dt = True


def generate_mesh(comm, lcar):
    radius_inner = 0.4292
    radius_outer = radius_inner + 1
    if comm.rank == 0:
        with pygmsh.geo.Geometry() as geom:
            circ_inner = geom.add_circle(
                [0.0, 0.0, 0.0],
                radius_inner,
                mesh_size=lcar,
                num_sections=12,
                compound=False, make_surface=False)

            geom.add_circle(
                [0.0, 0.0, 0.0],
                radius_outer,
                mesh_size=lcar,
                num_sections=24,
                compound=False,
                holes=[circ_inner.curve_loop]
                )

            mesh = geom.generate_mesh()
        mesh.prune_z_0()
        mesh.remove_lower_dimensional_cells()
        mesh.remove_orphaned_nodes()
        meshio.write_points_cells(
            "circ.xdmf", mesh.points,
            [("triangle", mesh.get_cells_type("triangle"))])

    mesh = dolfin.Mesh(comm)
    dolfin.XDMFFile("circ.xdmf").read(mesh)
    return mesh


def rotate(x, theta):
    """
    Rotate 2D vector x by an angle of theta using UFL.
    """
    R = ufl.as_matrix(((ufl.cos(theta), -ufl.sin(theta)),
                       (ufl.sin(theta), ufl.cos(theta))))
    return R * x


def theta_of_t(x, t_ufl):
    # r = ufl.sqrt(x[0]**2 + x[1]**2)
    return 1#-r*u0*(t_ufl + 0.25 * ufl.sin(2 * t_ufl))


class MMSCompressibleThermoChemBoussinesq:
    bounded = None

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        theta = theta_of_t(x, t_ufl)
        xc = dolfin.Constant((0.25, 0.0))
        sigma = dolfin.Constant(1e-1)
        x_rot = rotate(x, theta)
        phi_D = ufl.exp(-0.5 * (x_rot - xc) ** 2 / sigma ** 2)
        return phi_D

    def generate_T_d(self, mesh, t_ufl, kappa, sigma):
        a = "2*pow(sigma, 2)"
        b = "4*kappa"
        rsq_str = "x[0]*x[0] + x[1]*x[1]"
        T_exact = f"{a} / ({a} + {b} * t) " \
                  f"* exp(-({rsq_str})/({a} + {b} * t))"

        T_soln_bc = dolfin.Expression(
            T_exact, degree=6, sigma=sigma,
            kappa=kappa, domain=mesh, t=t_ufl)

        return T_soln_bc

    def run(self, comm, ElementClass, HeatClass, AdvectorClass, kappa_val=1e-3):
        directory = self.__class__.__name__
        run_name = ElementClass.__name__ + HeatClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Experiment properties
        Ra = dolfin.Constant(1.0)
        c_cfl = 1.0
        t_max = 0.1 if IS_TEST_ENV else 0.5

        # Record error functionals
        n_vals = [4, 8, 16, 32]
        u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        rhou_l2_error = np.zeros_like(n_vals, dtype=np.double)
        div_rhou_l2_error = np.zeros_like(n_vals, dtype=np.double)
        div_u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        T_l2_error = np.zeros_like(n_vals, dtype=np.double)
        final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)
        h_vals = np.zeros_like(n_vals, dtype=np.double)
        dofs = np.zeros_like(n_vals, dtype=np.double)
        error_in_time = [[] for _ in range(len(n_vals))]

        u_mag_max = 4.3

        for i, n_val in enumerate(n_vals):
            # FE mesh (write in serial and read in parallel since
            mesh = generate_mesh(comm, (n_val)**-1)

            dt_ufl = dolfin.Constant(0.0)
            t = 0.0
            t_ufl = dolfin.Constant(0.0)

            # Initialise Stokes and composition data structures
            # phistr = "atan2(x[1], x[0])"
            rstr = "sqrt(x[0]*x[0] + x[1]*x[1])"
            sin2atan2phi = "x[1]*x[1]/(x[0]*x[0] + x[1]*x[1] + DOLFIN_EPS)"
            t_func_str = "(pow(cos(t), 2) + 0.5)"
            t_func_ufl = ufl.cos(t_ufl)**2 + 0.5
            ustr = f"-x[1]*({sin2atan2phi} + 1)*{t_func_str}"
            vstr = f"x[0]*({sin2atan2phi} + 1)*{t_func_str}"
            u_soln = dolfin.Expression((ustr, vstr), degree=6, domain=mesh,
                                       t=t_ufl)

            # rhostr = f"-1.0/(cos(2*{phi}) - 3)"
            # rhostr = f"-{rstr}/(cos(2*{phistr}) - 3)"
            # rhostr = f"{rstr}/(3.0 - cos(2*{phistr}))"
            rhostr = f"{rstr}/(3.0 - (x[0]*x[0] - x[1]*x[1])/" \
                     f"(x[0]*x[0] + x[1]*x[1]))"
            rhobar = dolfin.Expression(rhostr, degree=6, domain=mesh)

            div_u_soln = dolfin.Expression(
                f"2*x[0]*x[1]/(x[0]*x[0] + x[1]*x[1])*{t_func_str}",
                degree=6, domain=mesh, t=t_ufl)

            rhou_soln = dolfin.Expression(
                (f"{rhostr}*{ustr}", f"{rhostr}*{vstr}"),
                degree=6, domain=mesh, t=t_ufl)

            element_cls = ElementClass()

            h_vals[i] = MPI.max(mesh.mpi_comm(), mesh.hmax())

            # Initialise particles. We need slots for [phi_p, T_p, dTdt_p].
            x = np.array([[0.0, 0.0]], dtype=np.float_)
            s = np.zeros((len(x), 1), dtype=np.float_)
            ptcls = leopart.particles(x, [s, s, s], mesh)

            # Stokes fields and function spaces
            W = element_cls.function_space(mesh)
            V = element_cls.velocity_function_space(mesh)
            u_vec = dolfin.Function(V)
            rhou_vec = dolfin.Function(V)

            U = element_cls.create_solution_variable(W)
            rhou = ufl.split(U[0])[0]

            # Initialise heat problem
            kappa = dolfin.Constant(kappa_val)
            sigma = dolfin.Constant(1.0)
            T_soln = self.generate_T_d(mesh, t_ufl, kappa, sigma)

            heat_property_idx = 1
            heat_cls = HeatClass(ptcls, rhou, dt_ufl, heat_property_idx)
            Sh_vec = heat_cls.function_space()
            T_vec = heat_cls.create_solution_variable(Sh_vec)
            Tstar_vec = heat_cls.create_solution_variable(Sh_vec)

            if isinstance(T_vec, (list, tuple)):
                T, Tbar = T_vec
            else:
                T = T_vec

            # Form the Stokes system momentum source and residual
            eta = dolfin.Constant(1.0)
            x = ufl.SpatialCoordinate(mesh)
            r = ufl.sqrt(x[0]*x[0] + x[1]*x[1])
            eps = dolfin.DOLFIN_EPS_LARGE
            f_expr = 4*ufl.as_vector((
                (5*x[0]**2*x[1] - 2*x[1]**3),
                (5*x[0]*x[1]**2 - 2*x[0]**3)
            ))/(3*r**4 + eps) * t_func_ufl
            f = f_expr
            f += Ra * (T - T_soln) * dolfin.Constant((0, 1))

            # Initial Stokes solve to get u(x, t=0) and p(x, t=0)
            model = geopart.stokes.compressible.CompressibleStokesModel(
                eta=eta, f=f, rho=rhobar)
            u = element_cls.get_velocity(U, model=model)

            # Set the initial system temperature and balance thermal
            # advection by the reaction term
            two_sigma_sq = dolfin.Constant(2*sigma**2)
            Q = dolfin.Expression(
                "4*a*k*(1-rho)*(a + 4*k*t - (x[0]*x[0] + x[1]*x[1])) / "
                "pow(a+4*k*t, 3) * exp(-(x[0]*x[0] + x[1]*x[1])/(a+4*k*t))",
                a=two_sigma_sq, k=kappa, rho=rhobar, t=t_ufl, degree=6
            )

            heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)
            heat_model.rho = rhobar
            heat_cls.T0_a.interpolate(T_soln)
            T.interpolate(T_soln)

            # Add particles to cells in a sweep. Don't need to bound the
            # sweep as we interpolate phi afterwards.
            n_parts_per_cell = 15*heat_cls.degree()
            if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
                ad = leopart.AddDelete(
                    ptcls, n_parts_per_cell, n_parts_per_cell,
                    [T, heat_cls.dTh0])
                ad.do_sweep()
                ptcls.interpolate(heat_cls.T0_a, heat_property_idx)
            else:
                ad = leopart.AddDelete(
                    ptcls, n_parts_per_cell, n_parts_per_cell, [T])
                ad.do_sweep()

            # Initialise the particle advector which requires solution of the
            # Stokes system
            def uh_accessor(step, dt):
                t_ufl.assign(t)
                if step == 0:
                    return u_vec._cpp_object
                if AdvectorClass is leopart.advect_rk2:
                    if step == 1:
                        t_ufl.assign(t + dt)
                        dt_ufl.assign(dt)
                elif AdvectorClass is leopart.advect_rk3:
                    if step == 1:
                        t_ufl.assign(t + 0.5 * dt)
                        dt_ufl.assign(0.5 * dt)
                    elif step == 2:
                        t_ufl.assign(t + 0.75 * dt)
                        dt_ufl.assign(0.75 * dt)
                elif AdvectorClass is leopart.advect_rk4:
                    if step in (1, 2):
                        t_ufl.assign(t + 0.5 * dt)
                        dt_ufl.assign(0.5 * dt)
                    elif step == 3:
                        t_ufl.assign(t + dt)
                        dt_ufl.assign(dt)

                # Projection of advective component
                heat_cls.project_advection(Tstar_vec, [], model=heat_model)

                # Solve the diffusive components
                if kappa_val > 0.0:
                    heat_cls.solve_diffusion(Sh_vec, T_vec, Tstar_vec,
                                             heat_mats, heat_bcs, heat_model)
                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                         assemble_lhs=False)

                momentum_assigner.assign(rhou_vec, element_cls.get_flux_soln(U))
                u_vec.interpolate(u_vec_func)
                # projector.project(u, u_vec)
                return u_vec._cpp_object

            ap = AdvectorClass(ptcls, V, uh_accessor, "open")

            # Setup Dirichlet BCs and the analytical velocity and
            # temperature solution
            ff = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1, 0)
            dolfin.CompiledSubDomain("on_boundary").mark(ff, 1)
            ds = ufl.Measure("ds", subdomain_data=ff)

            u_vec_func = dolfin.Expression(
                ("rhou/rhobar", "rhov/rhobar"),
                degree=6, rhou=U[0].sub(0).sub(0), rhov=U[0].sub(0).sub(1),
                rhobar=rhobar)

            weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), rhou_soln)]
            heat_bcs = [dolfin_dg.DGDirichletBC(ds(1), T_soln)]

            A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=True)

            momentum_assigner = dolfin.FunctionAssigner(
                rhou_vec.function_space(), element_cls.velocity_sub_space(W))
            momentum_assigner.assign(rhou_vec, element_cls.get_flux_soln(U))

            # Transfer the computed velocity and compute initial functionals
            # projector = geopart.projector.Projector(u_vec.function_space())
            # projector.project(u, u_vec)
            u_vec.interpolate(u_vec_func)

            hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())
            conservation0 = dolfin.assemble(rhobar*T * dx)
            conservation = abs(dolfin.assemble(rhobar*T * dx) - conservation0) \
                           / conservation0
            u_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            rhou_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            divu_l2 = dolfin.assemble(
                (ufl.div(u) - div_u_soln) ** 2 * dx) ** 0.5
            divrhou_l2 = dolfin.assemble(ufl.div(rhou) ** 2 * dx) ** 0.5
            T_l2 = dolfin.assemble((T - T_soln)**2 * dx)**0.5
            error_in_time[i].append(
                (t, rhou_l2, divrhou_l2, u_l2, divu_l2, T_l2, conservation))

            # Write initial functionals to file
            run_name_appendix = f"kappa{float(kappa)}_cfl{c_cfl}"
            fname_time = os.path.join(
                directory, f"{run_name}_{n_val}_{run_name_appendix}.txt")
            if comm.rank == 0:
                with open(fname_time, "w") as fi:
                    header = f"# h={hmin}, t, rhou_l2, divrhou_l2, u_l2, " \
                             f"divu_l2, T_l2, T_conservation\n"
                    fi.write(header)

            n_particles = MPI.sum(comm, len(ptcls.positions()))
            dolfin.info(f"Solving with {n_particles} particles, "
                        f"{W[0].dim()} stokes DoFs, "
                        f"and {T.function_space().dim()} heat DoFs")

            heat_mats = dolfin.PETScMatrix(mesh.mpi_comm()), \
                   dolfin.PETScVector(mesh.mpi_comm())

            ts = []
            T_l2s = []
            last_step = False
            for j in range(400):
                # Leopart indexes the first step as step = 1 in particle update
                step = j+1

                # Update dt and advect particles
                if use_fixed_dt:
                    dt_tentative = c_cfl * hmin / u_mag_max
                    dt = t_max / np.floor(t_max/dt_tentative)
                else:
                    dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)

                if not last_step and t + dt > t_max - dolfin.DOLFIN_EPS_LARGE:
                    dt = t_max - t
                    last_step = True

                dt_ufl.assign(dt)
                ad.do_sweep()
                ap.do_step(dt)
                dt_ufl.assign(dt)

                # Update time and compute the heat and Stokes system solution at
                # the next step
                t += dt
                t_ufl.assign(t)

                # Update heat
                heat_cls.project_advection(Tstar_vec, [], model=heat_model)
                if kappa_val > 0.0:
                    heat_cls.solve_diffusion(Sh_vec, T_vec, Tstar_vec,
                                             heat_mats, heat_bcs, heat_model)
                heat_cls.update_field_and_increment_particles(
                    T_vec, Tstar_vec, 0.5, step, dt
                )

                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                         assemble_lhs=False)
                momentum_assigner.assign(rhou_vec, element_cls.get_flux_soln(U))
                # projector.project(u, u_vec)
                u_vec.interpolate(u_vec_func)

                if step == 2:
                    # We don't immediately have enough information for the
                    # linear interpolation of dT/dt between time steps
                    # step 1: dTdt0 = 0, dTdt00 = 0 -> theta_L = 1.0
                    # step 2: dTdt0 =/= 0, dTdt00 = 0 -> theta_L = 1.0
                    # step 3: dTdt0 =/= 0, dTdt00 =/= 0 -> theta_L = 0.5
                    heat_cls.theta_L.assign(0.5)

                # Compute error functionals and write to file
                conservation = abs(dolfin.assemble(rhobar*T * dx) -
                                   conservation0) \
                    / conservation0
                u_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
                rhou_l2 = dolfin.assemble((rhou - rhou_soln) ** 2 * dx) ** 0.5
                divu_l2 = dolfin.assemble(
                    (ufl.div(u) - div_u_soln) ** 2 * dx) ** 0.5
                divrhou_l2 = dolfin.assemble(ufl.div(rhou) ** 2 * dx) ** 0.5
                T_l2 = dolfin.assemble((T - T_soln)**2 * dx)**0.5

                ts.append(t)
                T_l2s.append(T_l2)

                error_in_time[i].append(
                    (t, rhou_l2, divrhou_l2, u_l2, divu_l2, T_l2, conservation))

                # if n_val == 8:
                #     dolfin.XDMFFile("thermochem_T.xdmf").write_checkpoint(
                #         T, "T", time_step=t, append=j!=0
                #     )
                #     T_soln_interp = \
                #         dolfin.project((T_soln - T)**2, T.function_space())
                #     dolfin.XDMFFile("thermochem_Tsoln.xdmf").write_checkpoint(
                #         T_soln_interp, "Tsoln", time_step=t, append=j!=0
                #     )
                #     dolfin.XDMFFile("thermochem_u.xdmf").write_checkpoint(
                #         u_vec, "u_vec", time_step=t, append=j!=0
                #     )
                #     dolfin.XDMFFile("thermochem_rhou.xdmf").write_checkpoint(
                #         rhou_vec, "rhou_vec", time_step=t, append=j!=0
                #     )
                #     dolfin.XDMFFile("thermochem_p.xdmf").write_checkpoint(
                #         U[0].sub(1), "p", time_step=t, append=j!=0
                #     )
                #
                #     points_list = list(dolfin.Point(*pp)
                #                        for pp in ptcls.positions())
                #     particles_values = ptcls.get_property(heat_property_idx)
                #     dolfin.XDMFFile(
                #         os.path.join("./particlescomp/",
                #                      f"step{j:0>4d}.xdmf")) \
                #         .write(points_list, particles_values)

                update_msg = \
                    f"Timestep {j:>4}, Δt={dt:>.3e}, t={t:>.3e}, " \
                    f"‖rho uₕ - rho u‖₂={rhou_l2:>.3e}, " \
                    f"‖∇ ⋅ rho uₕ‖₂={divrhou_l2:>.3e}, " \
                    f"‖uₕ - u‖₂={u_l2:>.3e}, " \
                    f"‖∇ ⋅ (uₕ - u)‖₂={divu_l2:>.3e}," \
                    f"‖T - Tₕ‖₂={T_l2:>.3e}"
                dolfin.info(update_msg)

                if comm.rank == 0:
                    with open(fname_time, "a") as fi:
                        row = (t, rhou_l2, divrhou_l2, u_l2, divu_l2, T_l2,
                               conservation)
                        data_str = ", ".join(map(lambda ss: "%.16e" % ss, row))
                        fi.write(data_str + "\n")

                if last_step:
                    break

            u_l2_error[i] = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            rhou_l2_error[i] = dolfin.assemble(
                (rhou - rhou_soln) ** 2 * dx) ** 0.5
            dofs[i] = float(W[1].dim())
            final_mass_conservation[i] = conservation
            div_rhou_l2_error[i] = dolfin.assemble(
                ufl.div(rhou) ** 2 * dx) ** 0.5
            div_u_l2_error[i] = dolfin.assemble(
                (ufl.div(u) - div_u_soln) ** 2 * dx) ** 0.5
            T_l2_error[i] = dolfin.assemble((T - T_soln)**2 * dx)**0.5

        # Final error functionals and write to file
        u_l2_error = np.array(u_l2_error)
        rhou_l2_error = np.array(rhou_l2_error)
        h_vals = np.array(h_vals)
        dofs = np.array(dofs)
        final_mass_conservation = np.array(final_mass_conservation)
        div_u_l2_error = np.array(div_u_l2_error)
        T_l2_error = np.array(T_l2_error)

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        fname = os.path.join(directory, f"{run_name}_{run_name_appendix}.txt")
        data = np.vstack((
            h_vals, dofs, rhou_l2_error, div_rhou_l2_error,
            u_l2_error, div_u_l2_error, T_l2_error, final_mass_conservation))
        data = data.T

        if comm.rank == 0:
            header = "h, dofs, rhou_l2_error, div_rhou_l2_error, " \
                     "u_l2_error, div_u_l2_error, T_l2_error, " \
                     "final_mass_conservation"
            np.savetxt(fname, data, delimiter=",", header=header)

        # Compute and displace convergence rates
        h_rates = np.log(h_vals[1:] / h_vals[:-1])
        rates_rhou_l2 = np.log(rhou_l2_error[1:] / rhou_l2_error[:-1]) / h_rates
        rates_u_l2 = np.log(u_l2_error[1:] / u_l2_error[:-1]) / h_rates
        rates_divu_l2 = np.log(
            div_u_l2_error[1:] / div_u_l2_error[:-1]) / h_rates
        rates_T_l2 = np.log(T_l2_error[1:] / T_l2_error[:-1]) / h_rates

        dolfin.info(f"{directory}::{run_name} Velocity field L2 rates: "
                    f"{rates_u_l2}")
        dolfin.info(f"{directory}::{run_name} Compressibility L2 rates: "
                    f"{rates_divu_l2}")
        dolfin.info(f"{directory}::{run_name} Momentum field L2 rates: "
                    f"{rates_rhou_l2}")
        dolfin.info(f"{directory}::{run_name} Temperature field L2 rates: "
                    f"{rates_T_l2}")

        if IS_TEST_ENV:
            self.check_rates(rates_u_l2, element_cls, rates_T_l2, heat_cls)

    def check_rates(self,
                    rates_u_l2, stokes_cls,
                    rates_phi_l2, composition_cls,
                    rates_T_l2, heat_cls):
        assert np.all(rates_u_l2 > ((stokes_cls.degree() + 1) - 0.25))
        assert np.all(rates_phi_l2 > ((composition_cls.degree() + 1) - 0.25))
        assert np.all(rates_T_l2 > ((heat_cls.degree() + 1) - 0.25))


if __name__ == "__main__":
    # Compose and run a series of experiments

    HDG1 = geopart.stokes.compressible.HDGConservative
    HDG2 = geopart.stokes.compressible.HDG2Conservative
    # HeatHDG1l2 = geopart.energy.heat.HeatLeastSquaresNonlinearDiffusionHDG1
    # HeatHDG2l2 = geopart.energy.heat.HeatLeastSquaresNonlinearDiffusionHDG2
    HeatHDG1PDE = geopart.energy.heat.HeatPDEConstrainedNonlinearDiffusionHDG1
    HeatHDG2PDE = geopart.energy.heat.HeatPDEConstrainedNonlinearDiffusionHDG2

    rk2, rk3 = leopart.advect_rk2, leopart.advect_rk3

    cases = (
        # (HDG1, HeatHDG1l2, rk2),
        (HDG1, HeatHDG1PDE, rk2),
        # (HDG2, HeatHDG2l2, rk3),
        (HDG2, HeatHDG2PDE, rk3)
    )

    kappas = (0.0, 1e-3, 1e-1)

    c_cfl = (1.0,)

    for kappa in kappas:
        for case in cases:
            MMSCompressibleThermoChemBoussinesq().run(
                MPI.comm_world,
                case[0], case[1], case[2],
                kappa
            )

    # t_total = dolfin.Timer("geopart: total")
    # MMSCompressibleThermoChemBoussinesq().run(
    #     MPI.comm_world,
    #     geopart.stokes.compressible.HDGConservative,
    #     geopart.energy.heat.HeatLeastSquaresNonlinearDiffusionHDG1,
    #     # geopart.energy.heat.HeatPDEConstrainedNonlinearDiffusionHDG1,
    #     leopart.advect_rk2)
    # t_total.stop()

    # dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
