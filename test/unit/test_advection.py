import pytest

import numpy as np

import ufl
import dolfin

import leopart

import geopart.stokes.incompressible
import geopart.composition.incompressible
import geopart.projector


@pytest.mark.parametrize("AdvectorClass", [
    leopart.advect_particles,
    leopart.advect_rk2,
    leopart.advect_rk3,
    leopart.advect_rk4,
])
def test_advection(AdvectorClass):

    def rotate(x0, y0, theta):
        xp = x0 * np.cos(theta) - y0 * np.sin(theta)
        yp = x0 * np.sin(theta) + y0 * np.cos(theta)
        return xp, yp

    element_cls = geopart.stokes.incompressible.TaylorHood()

    r0 = 0.5
    xp = np.array([[r0, 0.0]], dtype=np.float_)

    period = 2.0 * np.pi

    n_vals = np.array([8, 16, 32], dtype=np.double)
    h_vals = np.zeros_like(n_vals, dtype=np.double)
    dtmins = np.zeros_like(n_vals, dtype=np.double)
    dtmaxs = np.zeros_like(n_vals, dtype=np.double)
    errors = np.zeros_like(n_vals, dtype=np.double)
    error_in_time = [[] for _ in range(len(n_vals))]
    c_cfl = 1.0

    for i, n_val in enumerate(n_vals):
        mesh = dolfin.RectangleMesh(dolfin.MPI.comm_world,
                                    dolfin.Point(-1, -1), dolfin.Point(1, 1),
                                    int(n_val), int(n_val), "left/right")

        # Define the variational (projection problem)
        V = element_cls.velocity_function_space(mesh)
        u_vec = dolfin.Function(V)

        u_soln = dolfin.Expression(("-x[1]*(pow(cos(t), 2) + 0.5)",
                                    "x[0]*(pow(cos(t), 2) + 0.5)"),
                                   t=0.0, degree=1, domain=mesh)
        proj = geopart.projector.Projector(V)
        proj.project(u_soln, u_vec)

        ptcls = leopart.particles(xp, [], mesh)

        # Particle advector
        hmin = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())

        def uh_accessor1(step, dt):
            if step == 0:
                u_soln.t = t
            else:
                assert False
            u_vec.assign(u_soln)
            return u_vec._cpp_object

        def uh_accessor2(step, dt):
            if step == 0:
                u_soln.t = t
            elif step == 1:
                u_soln.t = t + dt
            else:
                assert False
            u_vec.assign(u_soln)
            return u_vec._cpp_object

        def uh_accessor3(step, dt):
            if step == 0:
                u_soln.t = t
            elif step == 1:
                u_soln.t = t + 0.5 * dt
            elif step == 2:
                u_soln.t = t + 0.75 * dt
            else:
                assert False
            u_vec.assign(u_soln)
            return u_vec._cpp_object

        def uh_accessor4(step, dt):
            if step == 0:
                u_soln.t = t
            elif step in (1, 2):
                u_soln.t = t + 0.5 * dt
            elif step == 3:
                u_soln.t = t + dt
            else:
                assert False
            u_vec.assign(u_soln)
            return u_vec._cpp_object

        uh_accessor = {
            leopart.advect_particles: uh_accessor1,
            leopart.advect_rk2: uh_accessor2,
            leopart.advect_rk3: uh_accessor3,
            leopart.advect_rk4: uh_accessor4
        }

        ap = AdvectorClass(ptcls, V, uh_accessor[AdvectorClass], "closed")

        dt_min = 1e6
        dt_max = 0.0
        last_step = False
        t = 0.0

        positions = ptcls.get_property(0)
        positions = np.hstack(mesh.mpi_comm().allgather(positions)).reshape(
            (-1, 2))
        error_in_time[i].append((t, np.linalg.norm(positions[0] - xp[0], 2)))

        for j in range(50000):
            dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)

            if t + dt > period:
                dt = period - t
                last_step = True

            # Advect particles
            ap.do_step(dt)

            # Update dt and prepare new stokes solve
            t += dt
            dolfin.info("Timestep %d, dt = %.3e, t = %.3e"
                        % (j, float(dt), float(t)))

            theta_pos = t + 0.25 * np.sin(2.0 * t)
            xp_t = np.array(rotate(xp[0][0], xp[0][1], theta_pos))

            positions = ptcls.get_property(0)
            positions = np.hstack(mesh.mpi_comm().allgather(positions))\
                .reshape((-1, 2))
            error_in_time[i].append(
                (t, np.linalg.norm(positions[0] - xp_t, 2)))

            if last_step:
                break

        positions = ptcls.get_property(0)
        positions = np.hstack(mesh.mpi_comm().allgather(positions)).reshape(
            (-1, 2))
        errors[i] = np.linalg.norm(positions[0] - xp[0], 2)
        dtmins[i] = dt_min
        dtmaxs[i] = dt_max
        h_vals[i] = hmin

    cfl_rates = np.log(h_vals[:-1] / h_vals[1:])
    ratesu_l2 = np.log(errors[:-1] / errors[1:]) / cfl_rates

    rates = {
        leopart.advect_particles: 1.0,
        leopart.advect_rk2: 2.0,
        leopart.advect_rk3: 3.0,
        leopart.advect_rk4: 4.0
    }

    dolfin.info(str(ratesu_l2))
    assert np.all(ratesu_l2 > rates[AdvectorClass])


@pytest.mark.parametrize("AdvectorClass", [leopart.advect_rk2])
def test_field_advection(AdvectorClass):
    comm = dolfin.MPI.comm_world

    element_cls = geopart.stokes.incompressible.TaylorHood()

    u_soln = dolfin.Expression(("-x[1]*(pow(cos(t), 2) + 0.5)",
                                "x[0]*(pow(cos(t), 2) + 0.5)"),
                               t=0.0, degree=1)

    def rotate(x, theta):
        R = ufl.as_matrix(((ufl.cos(theta), -ufl.sin(theta)),
                           (ufl.sin(theta), ufl.cos(theta))))
        return R * x

    def generate_phi_d(mesh, t_ufl):
        theta = -(t_ufl + 0.25 * ufl.sin(2 * t_ufl))
        xy_t = rotate(ufl.SpatialCoordinate(mesh), theta)
        phi_D = ufl.sin(2 * ufl.pi * xy_t[0]) * ufl.sin(2 * ufl.pi * xy_t[1])
        return phi_D

    # Experiment properties
    t_max = np.pi / 8.0
    c_cfl = 1.0
    radius = 0.5

    # Record error functionals
    n_vals = np.array([4, 8, 16], dtype=np.double)
    h_vals = np.zeros_like(n_vals, dtype=np.double)
    errors_l2 = np.zeros_like(n_vals, dtype=np.double)
    final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)

    for i, n_val in enumerate(n_vals):
        # FE mesh (write in serial and read in parallel since
        # UnitDiscMesh does not generate in parallel)
        if comm.rank == 0:
            mesh = dolfin.UnitDiscMesh.create(
                dolfin.MPI.comm_self, int(n_val), 1, 2)
            dolfin.XDMFFile(dolfin.MPI.comm_self, "disc_mesh.xdmf").write(mesh)
        mesh = dolfin.Mesh(comm)
        dolfin.XDMFFile(comm, "disc_mesh.xdmf").read(mesh)
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
        composition_cls = geopart.composition.incompressible.PDEConstrainedDG1(
            ptcls, u_vec, dt_ufl, 1, bounded=None)
        Wh = composition_cls.function_space()
        phiD = generate_phi_d(mesh, t_ufl)
        phi = dolfin.project(phiD, Wh)
        composition_cls.update_field(phi)

        # Initialise particles with a specified number of particles per
        # cell
        npart = 25 * max(Wh.ufl_element().degree(), 1)
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
        hmin = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())
        h_vals[i] = hmin
        conservation0 = dolfin.assemble(phi * ufl.dx)

        last_step = False
        for j in range(50):
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
            dolfin.info("Timestep %d, dt = %.3e, t = %.3e"
                        % (j, float(dt), float(t)))

            composition_cls.update_field(phi)

            if last_step:
                break

        # Final error functionals and write to file
        errors_l2[i] = dolfin.assemble((phiD - phi) ** 2 * ufl.dx) ** 0.5
        final_mass_conservation[i] = \
            abs(dolfin.assemble(phi * ufl.dx) - conservation0)

    # Write final functionals to file
    h_rates = np.log(h_vals[:-1] / h_vals[1:])
    rates_phi_l2 = np.log(errors_l2[:-1] / errors_l2[1:]) / h_rates
    dolfin.info("Composition field L2 rates: %s" % str(rates_phi_l2))
    dolfin.info("Composition field l2 error: %s" % str(errors_l2))
    dolfin.info("Mass conservation error: %s" % str(final_mass_conservation))

    assert np.all(rates_phi_l2 > 1.98)
    assert np.all(final_mass_conservation < 1e-15)
