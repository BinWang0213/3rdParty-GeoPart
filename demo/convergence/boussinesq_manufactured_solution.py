import os

import dolfin
import dolfin_dg
import leopart
import numpy as np
import ufl
from dolfin import MPI
from ufl import dx

import geopart.composition.incompressible
import geopart.stokes
import geopart.stokes.incompressible

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False


def rotate(x, theta):
    """
    Rotate 2D vector x by an angle of theta using UFL.
    """
    R = ufl.as_matrix(((ufl.cos(theta), -ufl.sin(theta)),
                       (ufl.sin(theta), ufl.cos(theta))))
    return R * x


class MMSBoussinesq:
    bounded = None

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = -(t_ufl + 0.25 * ufl.sin(2 * t_ufl)) * r
        xc = dolfin.Constant((0.25, 0.0))
        sigma = dolfin.Constant(1e-1)
        x_rot = rotate(x, theta)
        phi_D = ufl.exp(-0.5 * (x_rot - xc) ** 2 / sigma**2)
        return phi_D

    def run(self, comm, ElementClass, CompositionClass, AdvectorClass):
        directory = self.__class__.__name__
        run_name = ElementClass.__name__ + CompositionClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Experiment properties
        Rb = dolfin.Constant(1.0)
        c_cfl = 1.0
        t_max = np.pi / 4.0 if IS_TEST_ENV else np.pi
        radius = 0.5

        # Record error functionals
        n_vals = [4, 8, 16]
        u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        div_u_l2_error = np.zeros_like(n_vals, dtype=np.double)
        phi_l2_error = np.zeros_like(n_vals, dtype=np.double)
        final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)
        h_vals = np.zeros_like(n_vals, dtype=np.double)
        dofs = np.zeros_like(n_vals, dtype=np.double)
        error_in_time = [[] for _ in range(len(n_vals))]

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

            # Initialise particles and composition field
            x = np.array([[0.0, 0.0]], dtype=np.float_)
            s = np.zeros((len(x), 1), dtype=np.float_)
            ptcls = leopart.particles(x, [s], mesh)

            phi_D = self.generate_phi_d(mesh, t_ufl)
            composition_cls = CompositionClass(ptcls, u_vec, dt_ufl, 1,
                                               bounded=self.bounded)
            Wh = composition_cls.function_space()
            phi = dolfin.project(phi_D, Wh)
            composition_cls.update_field(phi)

            # Add particles to cells in a sweep. Don't need to bound the
            # sweep as we interpolate phi afterwards.
            leopart.AddDelete(ptcls, 25, 30, [phi]).do_sweep()
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
                composition_cls.project_advection(phi)
                phi.vector().update_ghost_values()
                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)
                velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
                return u_vec._cpp_object

            ap = AdvectorClass(ptcls, V, uh_accessor, "open")

            # Setup Dirichlet BCs and the analytical velocity solution
            ff = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1, 0)
            dolfin.CompiledSubDomain("on_boundary").mark(ff, 1)
            ds = ufl.Measure("ds", subdomain_data=ff)

            u_soln = dolfin.Expression(
                ("-x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
                 "x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
                t=t_ufl, degree=6, domain=mesh)
            weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), u_soln)]

            # Form the Stokes system momentum source and residual
            eta = dolfin.Constant(1.0)
            x = ufl.SpatialCoordinate(mesh)
            r = ufl.sqrt(x[0]*x[0] + x[1]*x[1])
            f_expr = 3/r * ufl.as_vector((x[1], -x[0])) \
                * (ufl.cos(t_ufl)**2 + 0.5)
            f = Rb * (phi - phi_D) * dolfin.Constant((0, -1)) \
                + f_expr

            # Initial Stokes solve to get u(x, t=0) and p(x, t=0)
            U = element_cls.create_solution_variable(W)
            u, p = element_cls.get_velocity(U), element_cls.get_pressure(U)

            model = geopart.stokes.StokesModel(eta=eta, f=f)
            A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

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
            error_in_time[i].append((t, phi_l2, u_l2, conservation, divu_l2))

            # Write initial functionals to file
            fname_time = os.path.join(directory, f"{run_name}_{n_val}.txt")
            if comm.rank == 0:
                with open(fname_time, "w") as fi:
                    header = ", ".join([
                        "h=", str(hmin), "t", "error_l2_phi", "error_l2_u",
                        "mass_conservation", "divu_l2"])
                    fi.write("# " + header + "\n")

            n_particles = MPI.sum(comm, len(ptcls.positions()))
            dolfin.info("Solving with %d particles" % n_particles)

            last_step = False
            for j in range(400):
                # Update dt and advect particles
                dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
                if not last_step and t + dt > t_max:
                    dt = t_max - t
                    last_step = True
                dt_ufl.assign(dt)

                ap.do_step(dt)
                composition_cls.project_advection(phi)
                phi.vector().update_ghost_values()

                # Update time and compute the Stokes system solution at the
                # next step
                t += dt
                t_ufl.assign(t)
                element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

                # Compute error functionals and write to file
                conservation = abs(dolfin.assemble(phi * dx) - conservation0)
                phi_l2 = dolfin.assemble((phi_D - phi) ** 2 * dx) ** 0.5
                u_l2 = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
                divu_l2 = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5

                error_in_time[i].append(
                    (t, phi_l2, u_l2, conservation, divu_l2))

                update_msg = \
                    f"Timestep {j:>4}, Δt={dt:>.3e}, t={t:>.3e}, " \
                    f"‖φₕ - φ‖₂={phi_l2:>.3e}, ‖uₕ - u‖₂={u_l2:>.3e}, " \
                    f"|∫φₕ(x, t) - φₕ(x, 0) dx|={conservation:>.3e}, " \
                    f"‖∇ ⋅ uₕ‖₂={divu_l2:>.3e}"
                dolfin.info(update_msg)

                if comm.rank == 0:
                    with open(fname_time, "a") as fi:
                        row = (t, phi_l2, u_l2, conservation, divu_l2)
                        data_str = ", ".join(map(lambda ss: "%.16e" % ss, row))
                        fi.write(data_str + "\n")

                composition_cls.update_field(phi)

                if last_step or phi_l2 > 1e2:
                    break

            u_l2_error[i] = dolfin.assemble((u - u_soln) ** 2 * dx) ** 0.5
            phi_l2_error[i] = dolfin.assemble((phi_D - phi) ** 2 * dx) ** 0.5
            final_mass_conservation[i] = conservation
            dofs[i] = float(Wh.dim())
            div_u_l2_error[i] = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5

        # Final error functionals and write to file
        u_l2_error = np.array(u_l2_error)
        phi_l2_error = np.array(phi_l2_error)
        h_vals = np.array(h_vals)
        dofs = np.array(dofs)
        final_mass_conservation = np.array(final_mass_conservation)
        div_u_l2_error = np.array(div_u_l2_error)

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)

        fname = os.path.join(directory, run_name + ".txt")
        data = np.vstack((h_vals, dofs, u_l2_error, phi_l2_error,
                          final_mass_conservation, div_u_l2_error))
        data = data.T

        if comm.rank == 0:
            header = ", ".join(["h", "dofs", "error_l2_u", "error_l2_phi",
                                "mass_conservation", "divu_l2"])
            np.savetxt(fname, data, delimiter=",", header=header)

        # Compute and displace convergence rates
        h_rates = np.log(h_vals[1:] / h_vals[:-1])
        rates_u_l2 = np.log(u_l2_error[1:] / u_l2_error[:-1]) / h_rates
        rates_phi_l2 = np.log(phi_l2_error[1:] / phi_l2_error[:-1]) / h_rates
        dolfin.info(f"{directory} {run_name}: "
                    f"Composition field L2 rates: {rates_phi_l2}, "
                    f"Velocity field L2 rates: {rates_u_l2}")

        if IS_TEST_ENV:
            self.check_rates(rates_u_l2, element_cls,
                             rates_phi_l2, composition_cls)

    def check_rates(self,
                    rates_u_l2, stokes_cls,
                    rates_phi_l2, composition_cls):
        assert np.all(rates_u_l2 > ((stokes_cls.degree() + 1) - 0.25))
        assert np.all(rates_phi_l2 > 0.1)  # suboptimal

class MMSBoussinesqNonSmooth(MMSBoussinesq):
    bounded = None

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = -(t_ufl + 0.25 * ufl.sin(2 * t_ufl)) * r
        x_rot = rotate(x, theta)
        xtheta = ufl.atan_2(x_rot[1], x_rot[0])
        phi_D = ufl.conditional(ufl.gt(xtheta, 0.0), 1, 0)
        return phi_D

    def check_rates(self,
                    rates_u_l2, stokes_cls,
                    rates_phi_l2, composition_cls):
        assert np.all(rates_u_l2 > ((stokes_cls.degree() + 1) - 0.25))
        assert np.all(rates_phi_l2 > 0.1)  # suboptimal

class MMSBoussinesqNonSmoothBounded(MMSBoussinesqNonSmooth):
    bounded = (0.0, 1.0)


if __name__ == "__main__":
    # Compose and run a series of experiments
    experiment_classes = (
        MMSBoussinesq,
        MMSBoussinesqNonSmooth,
        MMSBoussinesqNonSmoothBounded,
    )

    for experiment in experiment_classes:
        experiment().run(
            MPI.comm_world,
            geopart.stokes.incompressible.HDG,
            geopart.composition.incompressible.PDEConstrainedDG1,
            leopart.advect_rk2)

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])