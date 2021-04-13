import os

import dolfin
import dolfin_dg
import leopart
import numpy as np
import ufl
from dolfin import MPI
from ufl import dx

import geopart.composition.incompressible
import geopart.energy.heat
import geopart.projector
import geopart.stokes.incompressible

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False

use_fixed_dt = False
u0 = dolfin.Constant(np.pi)


def rotate(x, theta):
    """
    Rotate 2D vector x by an angle of theta using UFL.
    """
    R = ufl.as_matrix(((ufl.cos(theta), -ufl.sin(theta)),
                       (ufl.sin(theta), ufl.cos(theta))))
    return R * x


def theta_of_t(x, t_ufl):
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    return -r*u0*(t_ufl + 0.25 * ufl.sin(2 * t_ufl))


class MMSThermoChemBoussinesq:
    bounded = None

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        theta = theta_of_t(x, t_ufl)
        xc = dolfin.Constant((0.25, 0.0))
        sigma = dolfin.Constant(1e-1)
        x_rot = rotate(x, theta)
        phi_D = ufl.exp(-0.5 * (x_rot - xc) ** 2 / sigma ** 2)
        return phi_D

    def generate_T_d(self, mesh, t_ufl, kappa):
        sigma = dolfin.Constant(0.1)
        x0, y0 = (0.0, 0.0)  # off centre analytical solution?!

        r = "sqrt(x[0]*x[0] + x[1]*x[1])"
        theta = f"u0*{r}*(t + 0.25 * sin(2 * t))"
        a = "2*pow(sigma, 2)"
        b = "4*kappa"
        x_rot = f"x[0]*cos({theta}) + x[1]*sin({theta})"
        y_rot = f"-x[0]*sin({theta}) + x[1]*cos({theta})"
        T_exact = f"{a} / ({a} + {b} * t) " \
                  f"* exp(-(pow({x_rot} - x0, 2) + pow({y_rot} - y0, 2))" \
                  f"/({a} + {b} * t))"

        T_soln_bc = dolfin.Expression(
            T_exact, degree=6, u0=u0, sigma=sigma,
            x0=x0, y0=y0, kappa=kappa, domain=mesh, t=t_ufl)

        return T_soln_bc

    def run(self, comm, ElementClass, HeatClass, CompositionClass,
            AdvectorClass):
        directory = self.__class__.__name__
        run_name = ElementClass.__name__ + CompositionClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Experiment properties
        Rb = dolfin.Constant(1.0)
        Ra = dolfin.Constant(1.0)
        c_cfl = 1.0
        t_max = 0.5 if IS_TEST_ENV else 2.0
        radius = 0.5

        # Record error functionals
        n_vals = [4, 8, 16]
        u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        div_u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        phi_l2_error = np.zeros_like(n_vals, dtype=np.double)
        T_l2_error = np.zeros_like(n_vals, dtype=np.double)
        final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)
        h_vals = np.zeros_like(n_vals, dtype=np.double)
        dofs = np.zeros_like(n_vals, dtype=np.double)
        error_in_time = [[] for _ in range(len(n_vals))]

        dt_fixed_vals = list(0.08 * 2 ** -a for a in range(len(n_vals)))

        for i, n_val in enumerate(n_vals):
            # FE mesh (write in serial and read in parallel since
            # UnitDiscMesh does not generate in parallel)
            mesh_name = f"{self.__class__.__name__}_mesh.xdmf"
            if comm.rank == 0:
                mesh = dolfin.UnitDiscMesh.create(
                    MPI.comm_self, int(n_val), 1, 2)
                mesh.coordinates()[:] *= radius
                dolfin.XDMFFile(MPI.comm_self, mesh_name).write(mesh)
            mesh = dolfin.Mesh(comm)
            dolfin.XDMFFile(comm, mesh_name).read(mesh)

            # Initialise Stokes and composition data structures
            element_cls = ElementClass()

            dt_ufl = dolfin.Constant(0.0)
            t = 0.0
            t_ufl = dolfin.Constant(0.0)

            h_vals[i] = MPI.min(mesh.mpi_comm(), mesh.hmin())

            # Stokes fields and function spaces
            W = element_cls.function_space(mesh)
            V = element_cls.velocity_function_space(mesh)
            u_vec = dolfin.Function(V)

            # Initialise particles. We need slots for [phi_p, T_p, dTdt_p].
            x = np.array([[0.0, 0.0]], dtype=np.float_)
            s = np.zeros((len(x), 1), dtype=np.float_)
            ptcls = leopart.particles(x, [s, s, s], mesh)

            # Initialise composition problem
            phi_D = self.generate_phi_d(mesh, t_ufl)
            phi_property_idx = 1
            composition_cls = CompositionClass(ptcls, u_vec, dt_ufl,
                                               phi_property_idx,
                                               bounded=self.bounded)
            Wh = composition_cls.function_space()
            phi = dolfin.project(phi_D, Wh)
            composition_cls.update_field(phi)

            u_soln = dolfin.Expression(
                ("-u0*x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
                 "u0*x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
                t=t_ufl, degree=6, domain=mesh, u0=u0)

            # Initialise heat problem
            kappa = dolfin.Constant(1e-3)
            T_soln = self.generate_T_d(mesh, t_ufl, kappa)

            heat_property_idx = 2
            heat_cls = HeatClass(ptcls, u_vec, dt_ufl, heat_property_idx)
            Sh_vec = heat_cls.function_space()
            T_vec = heat_cls.create_solution_variable(Sh_vec)
            Tstar_vec = heat_cls.create_solution_variable(Sh_vec)

            if isinstance(T_vec, (list, tuple)):
                T, Tbar = T_vec
            else:
                T = T_vec

            # Set the initial system temperature and balance thermal
            # advection by the reaction term
            Q = dolfin.Constant(0.0)
            heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)
            heat_cls.T0_a.interpolate(T_soln)
            T.interpolate(T_soln)

            # Add particles to cells in a sweep. Don't need to bound the
            # sweep as we interpolate phi afterwards.
            n_parts_per_cell = 25*heat_cls.degree()
            if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
                leopart.AddDelete(ptcls, n_parts_per_cell, n_parts_per_cell,
                                  [phi, T, heat_cls.dTh0]).do_sweep()
                ptcls.interpolate(heat_cls.T0_a, 2)
            else:
                leopart.AddDelete(ptcls, n_parts_per_cell, n_parts_per_cell,
                                  [phi]).do_sweep()
            ptcls.interpolate(phi, 1)

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
                composition_cls.project_advection(phi)
                heat_cls.project_advection(Tstar_vec, [])

                # Solve the diffusive components
                heat_cls.solve_diffusion(Sh_vec, T_vec, Tstar_vec, heat_mats,
                                    heat_bcs, heat_model)
                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                         assemble_lhs=False)

                velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
                return u_vec._cpp_object

            ap = AdvectorClass(ptcls, V, uh_accessor, "open")

            # Setup Dirichlet BCs and the analytical velocity and
            # temperature solution
            ff = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1, 0)
            dolfin.CompiledSubDomain("on_boundary").mark(ff, 1)
            ds = ufl.Measure("ds", subdomain_data=ff)

            weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), u_soln)]
            heat_bcs = [dolfin_dg.DGDirichletBC(ds(1), T_soln)]

            # Form the Stokes system momentum source and residual
            eta = dolfin.Constant(1.0)
            x = ufl.SpatialCoordinate(mesh)
            r = ufl.sqrt(x[0]*x[0] + x[1]*x[1])
            f_expr = 3*u0/r * ufl.as_vector((x[1], -x[0])) \
                * (ufl.cos(t_ufl)**2 + 0.5)
            f = f_expr
            f += Rb * (phi - phi_D) * dolfin.Constant((0, -1))
            f += Ra * (T - T_soln) * dolfin.Constant((0, 1))

            # Initial Stokes solve to get u(x, t=0) and p(x, t=0)
            U = element_cls.create_solution_variable(W)
            u, p = element_cls.get_velocity(U), element_cls.get_pressure(U)

            model = geopart.stokes.StokesModel(eta=eta, f=f)
            A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=True)

            # Transfer the computed velocity and compute initial functionals
            velocity_assigner = dolfin.FunctionAssigner(
                u_vec.function_space(), element_cls.velocity_sub_space(W))
            velocity_assigner.assign(u_vec, element_cls.get_velocity(U))

            hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())
            conservation0 = dolfin.assemble(phi * dx)
            conservation = abs(dolfin.assemble(phi * dx) - conservation0)
            phi_l2 = dolfin.assemble((phi_D - phi) ** 2 * dx) ** 0.5
            u_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            divu_l2 = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5
            T_l2 = dolfin.assemble((T - T_soln)**2 * dx)**0.5
            error_in_time[i].append(
                (t, phi_l2, u_l2, conservation, divu_l2, T_l2))

            # Write initial functionals to file
            fname_time = os.path.join(directory,
                                      run_name + "_" + str(n_val) + ".txt")
            if comm.rank == 0:
                with open(fname_time, "w") as fi:
                    header = ", ".join([
                        "h=", str(hmin), "t", "error_l2_phi", "error_l2_u",
                        "mass_conservation", "divu_l2"])
                    fi.write("# " + header + "\n")

            n_particles = MPI.sum(comm, len(ptcls.positions()))
            dolfin.info("Solving with %d particles, %d stokes DoFs, "
                        "%d composition DoFs and %d heat DoFs" %
                        (n_particles, u.function_space().dim(),
                         phi.function_space().dim(),
                         T.function_space().dim()))

            heat_mats = dolfin.PETScMatrix(mesh.mpi_comm()), \
                   dolfin.PETScVector(mesh.mpi_comm())

            projector = geopart.projector.Projector(Wh, dx)
            phi_exact = dolfin.Function(Wh)

            ts = []
            T_l2s = []
            last_step = False
            for j in range(400):
                step = j+1

                # Update dt and advect particles
                if use_fixed_dt:
                    dt = dt_fixed_vals[i]
                else:
                    dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)

                if not last_step and t + dt > t_max:
                    dt = t_max - t
                    last_step = True

                dt_ufl.assign(dt)
                ap.do_step(dt)
                dt_ufl.assign(dt)
                composition_cls.project_advection(phi)
                composition_cls.update_field(phi)

                # Update time and compute the heat and Stokes system solution at
                # the next step
                t += dt
                t_ufl.assign(t)

                # Update heat
                heat_cls.project_advection(Tstar_vec, [])
                heat_cls.solve_diffusion(Sh_vec, T_vec, Tstar_vec, heat_mats,
                                    heat_bcs, heat_model)
                heat_cls.update_field_and_increment_particles(
                    T_vec, Tstar_vec, 0.5, step, dt
                )

                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                         assemble_lhs=False)

                if step == 2:
                    # We don't immediately have enough information for the
                    # linear interpolation of dT/dt between time steps
                    # step 1: dTdt0 = 0, dTdt00 = 0 -> theta_L = 1.0
                    # step 2: dTdt0 =/= 0, dTdt00 = 0 -> theta_L = 1.0
                    # step 3: dTdt0 =/= 0, dTdt00 =/= 0 -> theta_L = 0.5
                    heat_cls.theta_L.assign(0.5)

                # Compute error functionals and write to file
                conservation = abs(dolfin.assemble(phi * dx) - conservation0)
                phi_l2 = dolfin.assemble((phi_D - phi) ** 2 * dx) ** 0.5
                u_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
                divu_l2 = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5
                T_l2 = dolfin.assemble((T - T_soln)**2 * dx)**0.5

                ts.append(t)
                T_l2s.append(T_l2)

                error_in_time[i].append(
                    (t, phi_l2, u_l2, conservation, divu_l2, T_l2))

                if n_val == 16:
                    dolfin.XDMFFile("thermochem_phi.xdmf").write_checkpoint(
                        phi, "phi", time_step=t, append=j!=0
                    )
                    projector.project(phi_D, phi_exact)
                    dolfin.XDMFFile(
                        "thermochem_phi_exact.xdmf").write_checkpoint(
                        phi_exact, "phi_exact", time_step=t, append=j!=0
                    )
                    dolfin.XDMFFile("thermochem_T.xdmf").write_checkpoint(
                        T, "T", time_step=t, append=j!=0
                    )
                    dolfin.XDMFFile("thermochem_u.xdmf").write_checkpoint(
                        u_vec, "u_vec", time_step=t, append=j!=0
                    )

                update_msg = \
                    f"Timestep {j:>4}, Δt={dt:>.3e}, t={t:>.3e}, " \
                    f"‖φₕ - φ‖₂={phi_l2:>.3e}, ‖uₕ - u‖₂={u_l2:>.3e}, " \
                    f"|∫φₕ(x, t) - φₕ(x, 0) dx|={conservation:>.3e}, " \
                    f"‖∇ ⋅ uₕ‖₂={divu_l2:>.3e}, ‖T - Tₕ‖₂={T_l2:>.3e}"
                dolfin.info(update_msg)

                if comm.rank == 0:
                    with open(fname_time, "a") as fi:
                        row = (t, phi_l2, u_l2, conservation, divu_l2)
                        data_str = ", ".join(map(lambda ss: "%.16e" % ss, row))
                        fi.write(data_str + "\n")

                if last_step or phi_l2 > 1e2:
                    break

            u_l2_error[i] = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            phi_l2_error[i] = dolfin.assemble((phi_D - phi) ** 2 * dx) ** 0.5
            final_mass_conservation[i] = conservation
            dofs[i] = float(Wh.dim())
            div_u_l2_error[i] = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5
            T_l2_error[i] = dolfin.assemble((T - T_soln)**2 * dx)**0.5

        # Final error functionals and write to file
        u_l2_error = np.array(u_l2_error)
        phi_l2_error = np.array(phi_l2_error)
        h_vals = np.array(h_vals)
        dofs = np.array(dofs)
        final_mass_conservation = np.array(final_mass_conservation)
        div_u_l2_error = np.array(div_u_l2_error)
        T_l2_error = np.array(T_l2_error)

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        fname = os.path.join(directory, run_name + ".txt")
        data = np.vstack((h_vals, dofs, u_l2_error, phi_l2_error,
                          final_mass_conservation, div_u_l2_error,
                          T_l2_error))
        data = data.T

        if comm.rank == 0:
            header = ", ".join(["h", "dofs", "error_l2_u", "error_l2_phi",
                                "mass_conservation", "divu_l2", "T_l2"])
            np.savetxt(fname, data, delimiter=",", header=header)

        # Compute and displace convergence rates
        h_rates = np.log(h_vals[1:] / h_vals[:-1])
        rates_u_l2 = np.log(u_l2_error[1:] / u_l2_error[:-1]) / h_rates
        rates_phi_l2 = np.log(phi_l2_error[1:] / phi_l2_error[:-1]) / h_rates
        rates_T_l2 = np.log(T_l2_error[1:] / T_l2_error[:-1]) / h_rates

        dolfin.info(f"{directory}::{run_name} Composition field L2 rates: "
                    f"{rates_phi_l2}")
        dolfin.info(f"{directory}::{run_name} Velocity field L2 rates: "
                    f"{rates_u_l2}")
        dolfin.info(f"{directory}::{run_name} Temperature field L2 rates: "
                    f"{rates_T_l2}")

        if IS_TEST_ENV:
            self.check_rates(rates_u_l2, element_cls, rates_phi_l2,
                             composition_cls, rates_T_l2, heat_cls)

    def check_rates(self,
                    rates_u_l2, stokes_cls,
                    rates_phi_l2, composition_cls,
                    rates_T_l2, heat_cls):
        assert np.all(rates_u_l2 > ((stokes_cls.degree() + 1) - 0.25))
        assert np.all(rates_phi_l2 > ((composition_cls.degree() + 1) - 0.25))
        assert np.all(rates_T_l2 > ((heat_cls.degree() + 1) - 0.25))


class MMSThermoChemBoussinesqNonSmooth(MMSThermoChemBoussinesq):
    bounded = None

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = - u0 * (t_ufl + 0.25 * ufl.sin(2 * t_ufl)) * r
        x_rot = rotate(x, theta)
        xtheta = ufl.atan_2(x_rot[1], x_rot[0])
        phi_D = ufl.conditional(ufl.gt(xtheta, 0.0), 1, 0)
        return phi_D

    def check_rates(self,
                    rates_u_l2, stokes_cls,
                    rates_phi_l2, composition_cls,
                    rates_T_l2, heat_cls):
        assert np.all(rates_u_l2 > ((stokes_cls.degree() + 1) - 0.25))
        assert np.all(rates_phi_l2 > 0.4)  # suboptimal
        assert np.all(rates_T_l2 > ((heat_cls.degree() + 1) - 0.25))


class MMSThermoChemBoussinesqNonSmoothBounded(MMSThermoChemBoussinesqNonSmooth):
    bounded = (0.0, 1.0)


if __name__ == "__main__":
    # Compose and run a series of experiments
    experiment_classes = (
        MMSThermoChemBoussinesq,
        # MMSThermoChemBoussinesqNonSmooth,
        # MMSThermoChemBoussinesqNonSmoothBounded,
    )

    t_total = dolfin.Timer("geopart: total")
    for experiment in experiment_classes:
        experiment().run(
            MPI.comm_world,
            geopart.stokes.incompressible.HDG,
            geopart.energy.heat.HeatPDEConstrainedHDG1,
            geopart.composition.incompressible.PDEConstrainedDG1,
            leopart.advect_rk2)
    t_total.stop()

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
