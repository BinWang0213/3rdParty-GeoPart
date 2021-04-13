import os

import dolfin
import leopart
import numpy as np
import ufl
from dolfin import MPI
from ufl import dx

import geopart.composition.incompressible
import geopart.projector
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


def n_part_per_cell(k):
    """
    Number of particles used per cell, where k is the polynomial degree of
    the composition field finite element function space.
    """
    return (k + 1) * 25


is_linear_velocity = True
# The analytical velocity field used in all experiments
if is_linear_velocity:
    u_soln = dolfin.Expression(("-x[1]*(pow(cos(t), 2) + 0.5)",
                                "x[0]*(pow(cos(t), 2) + 0.5)"),
                               t=0.0, degree=1)
else:
    u_soln = dolfin.Expression(
        ("-u0*x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
         "u0*x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
        t=0.0, degree=6, u0=1.0)


def theta_of_t(x, t_ufl):
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    theta = -(t_ufl + 0.25 * ufl.sin(2 * t_ufl))
    return theta if is_linear_velocity else r*theta


class SmoothFieldAdvectionCircle:
    bounded = False

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        theta = theta_of_t(x, t_ufl)
        xc = dolfin.Constant((0.25, 0.0))
        sigma = dolfin.Constant(5e-1)
        x_rot = rotate(x, theta)
        phi_D = ufl.exp(-0.5 * (x_rot - xc) ** 2 / sigma)
        return phi_D

    def run(self, comm, ElementClass, CompositionClass, AdvectorClass,
            write_composition_field=False):
        # Run directory and name
        directory = self.__class__.__name__
        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
        run_name = ElementClass.__name__ \
            + CompositionClass.__name__ \
            + AdvectorClass.__name__

        element_cls = ElementClass()

        # Experiment properties
        t_max = np.pi / 4.0
        c_cfl = 1.0
        radius = 2.0

        # Record error functionals
        n_vals = np.array([4, 8, 16], dtype=np.double)
        h_vals = np.zeros_like(n_vals, dtype=np.double)
        errors_l2 = np.zeros_like(n_vals, dtype=np.double)
        final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)
        dofs = np.zeros_like(n_vals, dtype=np.double)
        error_in_time = [[] for _ in range(len(n_vals))]

        for i, n_val in enumerate(n_vals):
            # FE mesh (write in serial and read in parallel since
            # UnitDiscMesh does not generate in parallel)
            mesh_name = f"{self.__class__.__name__}_mesh.xdmf"
            if comm.rank == 0:
                mesh = dolfin.UnitDiscMesh.create(
                    MPI.comm_self, int(n_val), 1, 2)
                dolfin.XDMFFile(MPI.comm_self, mesh_name).write(mesh)
            mesh = dolfin.Mesh(comm)
            dolfin.XDMFFile(comm, mesh_name).read(mesh)
            mesh.coordinates()[:] *= radius

            # Velocity function space and solution
            V = element_cls.velocity_function_space(mesh)
            u_vec = dolfin.Function(V)
            u_vec.assign(u_soln)

            # Time variables
            t = 0.0
            dt_ufl = dolfin.Constant(0.0)
            t_ufl = dolfin.Constant(t)

            # Generate leopart.particles with single particle
            xp = np.array([[0.0, 0.0]], dtype=np.float_)
            xs = [np.zeros((len(xp), 1), dtype=np.float_)]
            ptcls = leopart.particles(xp, xs, mesh)

            # Compose analytical composition field and its approximation
            composition_cls = CompositionClass(ptcls, u_vec, dt_ufl, 1,
                                               bounded=self.bounded)
            Wh = composition_cls.function_space()
            phiD = self.generate_phi_d(mesh, t_ufl)
            phi = dolfin.project(phiD, Wh)
            composition_cls.update_field(phi)

            # Initialise particles with a specified number of particles per
            # cell
            npart = n_part_per_cell(Wh.ufl_element().degree())
            leopart.AddDelete(ptcls, npart, npart, [phi]).do_sweep()
            ptcls.interpolate(phi, 1)

            # Particle advector
            def uh_accessor(step, dt):
                if AdvectorClass is leopart.advect_particles:
                    if step == 0:
                        u_soln.t = t
                elif AdvectorClass is leopart.advect_rk2:
                    if step == 0:
                        u_soln.t = t
                    elif step == 1:
                        u_soln.t = t + dt
                elif AdvectorClass is leopart.advect_rk3:
                    if step == 0:
                        u_soln.t = t
                    elif step == 1:
                        u_soln.t = t + 0.5 * dt
                    elif step == 2:
                        u_soln.t = t + 0.75 * dt
                elif AdvectorClass is leopart.advect_rk4:
                    if step == 0:
                        u_soln.t = t
                    elif step in (1, 2):
                        u_soln.t = t + 0.5 * dt
                    elif step == 3:
                        u_soln.t = t + dt
                u_vec.assign(u_soln)
                return u_vec._cpp_object

            ap = AdvectorClass(ptcls, V, uh_accessor, "open")

            # Record initial states
            hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())
            h_vals[i] = hmin
            conservation0 = dolfin.assemble(phi * dx)
            conservation = abs(dolfin.assemble(phi * dx) - conservation0)
            phi_l2 = dolfin.assemble((phiD - phi) ** 2 * dx) ** 0.5
            error_in_time[i].append((t, phi_l2, conservation))

            V_exact = dolfin.FunctionSpace(mesh, "DG", 2)
            projector = geopart.projector.Projector(V_exact, dx)
            phi_exact = dolfin.Function(V_exact)

            # Write first step to file
            xdmf_file_name = os.path.join(
                directory,
                "%s_phi_%d.xdmf" % (run_name, n_val))
            xdmf_file_name_exact = os.path.join(
                directory,
                "%s_phi_exact_%d.xdmf" % (run_name, n_val))
            if write_composition_field:
                dolfin.XDMFFile(xdmf_file_name).write_checkpoint(
                    phi, "phi", time_step=t, append=False)
                projector.project(phiD, phi_exact)
                dolfin.XDMFFile(xdmf_file_name_exact).write_checkpoint(
                    phi_exact, "phi_exact", time_step=t, append=False)


            last_step = False
            for j in range(500):
                # Update dt, advect particles and project to field
                dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
                if t + dt > t_max:
                    dt = t_max - t
                    last_step = True
                dt_ufl.assign(dt)
                ap.do_step(dt)
                composition_cls.project_advection(phi)

                # Update time and compute error functionals
                t += dt
                t_ufl.assign(t)

                conservation = abs(dolfin.assemble(phi * dx) - conservation0)
                phi_l2 = dolfin.assemble((phiD - phi) ** 2 * dx) ** 0.5
                error_in_time[i].append((t, phi_l2, conservation))

                dolfin.info(f"Timestep {j:>4d}, dt={dt:<.3e}, t={t:<.3e}, "
                            f"phi l2 error={phi_l2:<.3e}, "
                            f"conservation={conservation:<.3e}")

                # Output field for visualisation
                if write_composition_field:
                    dolfin.XDMFFile(xdmf_file_name).write_checkpoint(
                        phi, "phi", time_step=t, append=True)
                    projector.project(phiD, phi_exact)
                    dolfin.XDMFFile(xdmf_file_name_exact).write_checkpoint(
                        phi_exact, "phi_exact", time_step=t, append=True)

                # Store phi into phi0
                composition_cls.update_field(phi)

                if last_step:
                    break

            # Final error functionals and write to file
            errors_l2[i] = dolfin.assemble((phiD - phi) ** 2 * dx) ** 0.5
            final_mass_conservation[i] = conservation
            dofs[i] = float(Wh.dim())

            fname = os.path.join(directory,
                                 run_name + "_" + str(n_val) + ".txt")
            output_data = np.array(error_in_time[i], dtype=np.double)
            if comm.rank == 0:
                header = ", ".join(["{h=" + str(hmin) + "}", "t",
                                    "error_l2_phi", "mass_conservation"])
                np.savetxt(fname, output_data, delimiter=",", header=header)

        # Write final functionals to file
        h_rates = np.log(h_vals[:-1] / h_vals[1:])
        rates_phi_l2 = np.log(errors_l2[:-1] / errors_l2[1:]) / h_rates
        dolfin.info(directory + run_name + ": Composition field L2 rates: %s"
                    % str(rates_phi_l2))
        dolfin.info(directory + run_name + ": Composition field l2 error: %s"
                    % str(errors_l2))

        fname = os.path.join(directory, run_name + ".txt")
        data = np.vstack((h_vals, dofs, errors_l2, final_mass_conservation))
        data = data.T
        if comm.rank == 0:
            header = ", ".join(["h", "dofs", "error_l2_phi",
                                "mass_conservation"])
            np.savetxt(fname, data, delimiter=",", header=header)

        if IS_TEST_ENV:
            self.check_rates(rates_phi_l2, composition_cls)

    def check_rates(self, rates_phi_l2, composition_cls):
        assert np.all(rates_phi_l2 > composition_cls.degree() + 1 - 0.25)


class NonSmoothFieldAdvectionCircle(SmoothFieldAdvectionCircle):
    bounded = False

    def generate_phi_d(self, mesh, t_ufl):
        x = ufl.SpatialCoordinate(mesh)
        theta = theta_of_t(x, t_ufl)
        x_rot = rotate(x, theta)
        xtheta = ufl.atan_2(x_rot[1], x_rot[0])
        phi_D = ufl.conditional(ufl.gt(xtheta, 0.0), 1, 0)
        return phi_D

    def check_rates(self, rates_phi_l2, composition_cls):
        pass  # Suboptimal / no convergence at this resolution

class NonSmoothFieldAdvectionCircleBounded(NonSmoothFieldAdvectionCircle):
    bounded = (0.0, 1.0)


if __name__ == "__main__":
    # Compose and run a series of experiments
    experiment_classes = (
        SmoothFieldAdvectionCircle,
        NonSmoothFieldAdvectionCircle,
        NonSmoothFieldAdvectionCircleBounded
    )

    for experiment in experiment_classes:
        experiment().run(
            MPI.comm_world,
            geopart.stokes.incompressible.HDG,
            geopart.composition.incompressible.PDEConstrainedDG1,
            leopart.advect_rk2,
            write_composition_field=False)
