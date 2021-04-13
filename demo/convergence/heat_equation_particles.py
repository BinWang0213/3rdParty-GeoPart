import os

import dolfin
import leopart
import numpy as np
import ufl

import geopart.energy.heat
import geopart.projector
import geopart.stokes.incompressible

comm = dolfin.MPI.comm_world

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False

element_cls = geopart.stokes.incompressible.HDG()
AdvectorClass = leopart.advect_rk3
HeatClass = geopart.energy.heat.HeatPDEConstrainedHDG2

# Governing parameters:
#   kappa = 0 & u nonzero => pure advection
#   kappa nonzero & u nonzero => advection-diffusion
#   kappa nonzero & u = 0 => pure diffusion
use_fixed_dt = True
kappa = dolfin.Constant(1e-3)
u0 = dolfin.Constant(np.pi)

# Experiment properties
t_max = 0.4 if IS_TEST_ENV else 2.0
c_cfl = 1.0
radius = 0.5
is_linear_velocity = True
write_output = False


def rotate(x, theta):
    R = ufl.as_matrix(((ufl.cos(u0*theta), -ufl.sin(u0*theta)),
                       (ufl.sin(u0*theta), ufl.cos(u0*theta))))
    return R * x


def generate_T_d(mesh, t_ufl):
    sigma = dolfin.Constant(0.1)
    x0, y0 = (-0.15, 0.0) if is_linear_velocity else (0.0, 0.0)

    theta = "Uh*(t + 0.25 * sin(2 * t))"
    if not is_linear_velocity:
        r = "sqrt(x[0]*x[0] + x[1]*x[1])"
        theta = r + "*" + theta
    a = "2*pow(sigma, 2)"
    b = "4*kappa"
    x_rot = f"x[0]*cos({theta}) + x[1]*sin({theta})"
    y_rot = f"-x[0]*sin({theta}) + x[1]*cos({theta})"
    T_exact = f"{a} / ({a} + {b} * t) " \
              f"* exp(-(pow({x_rot} - x0, 2) + pow({y_rot} - y0, 2))" \
              f"/({a} + {b} * t))"

    T_soln_bc = dolfin.Expression(
        T_exact, degree=6, Uh=u0, sigma=sigma,
        x0=x0, y0=y0, kappa=kappa, domain=mesh, t=t_ufl)

    return T_soln_bc

# Record error functionals
n_vals = np.array([4, 8, 16], dtype=np.double)
h_vals = np.zeros_like(n_vals, dtype=np.double)
n_parts_list = np.zeros_like(n_vals, dtype=np.double)
errors_l2 = np.zeros_like(n_vals, dtype=np.double)
final_mass_conservation = np.zeros_like(n_vals, dtype=np.double)
dt_fixed_vals = list(0.08 * 2 ** -a for a in range(len(n_vals)))


for i, n_val in enumerate(n_vals):
    # FE mesh (write in serial and read in parallel since
    # UnitDiscMesh does not generate in parallel)
    mesh_name = "heat_eqn_particles_mesh.xdmf"
    if comm.rank == 0:
        mesh = dolfin.UnitDiscMesh.create(
            dolfin.MPI.comm_self, int(n_val), 1, 2)
        dolfin.XDMFFile(dolfin.MPI.comm_self, mesh_name).write(mesh)
    mesh = dolfin.Mesh(comm)
    dolfin.XDMFFile(comm, mesh_name).read(mesh)
    mesh.coordinates()[:] *= radius

    if is_linear_velocity:
        u_soln = dolfin.Expression(("-u0*x[1]*(pow(cos(t), 2) + 0.5)",
                                    "u0*x[0]*(pow(cos(t), 2) + 0.5)"),
                                   u0=u0, t=0.0, degree=1)
    else:
        u_soln = dolfin.Expression(
            ("-u0*x[1]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)",
             "u0*x[0]*sqrt(x[0]*x[0] + x[1]*x[1])*(pow(cos(t), 2) + 0.5)"),
            t=0.0, degree=6, u0=u0)

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
    xs = [np.zeros((len(xp), 1), dtype=np.float_),
          np.zeros((len(xp), 1), dtype=np.float_)]
    ptcls = leopart.particles(xp, xs, mesh)

    # Compose analytical composition field and its approximation
    heat_cls = HeatClass(ptcls, u_vec, dt_ufl, 1)
    Wh = heat_cls.function_space()
    Tstar, Tstarbar = heat_cls.create_solution_variable(Wh)
    T, Tbar = heat_cls.create_solution_variable(Wh)
    T_D_bc = generate_T_d(mesh, t_ufl)

    # Although projection is a better inital approximation by Cea's Lemma,
    # it will not satisfy the BCs
    heat_cls.T0_a.assign(T_D_bc)

    # Initialise particles with a specified number of particles per
    # cell
    npart = 25 * max(Wh[0].ufl_element().degree(), 1)
    ad = leopart.AddDelete(ptcls, npart, npart, [T])
    ad.do_sweep()

    # Construct the projection of the exact solution which will be used to
    # interpolate onto the particles as their initial condition
    T_exact = dolfin.Function(Wh[0])
    exact_projector = geopart.projector.Projector(
        T_exact.function_space(), ufl.dx)
    exact_projector.project(T_D_bc, T_exact)
    ptcls.interpolate(T_exact, 1)

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

    # Construct the heat model with homogeneous source term
    f = dolfin.Constant(0.0)
    heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=f)

    # Record initial states
    hmin = dolfin.MPI.min(mesh.mpi_comm(), mesh.hmin())
    h_vals[i] = hmin
    n_parts_list[i] = ptcls.number_of_particles()
    conservation0 = dolfin.assemble(heat_cls.T0_a * ufl.dx)

    if write_output:
        dolfin.XDMFFile("T_debug.xdmf").write_checkpoint(
            heat_cls.T0_a, "T", time_step=0.0, append=False)
        dolfin.XDMFFile("T_exact.xdmf").write_checkpoint(
            T_exact, "T_exact", time_step=0.0, append=False)

    # The boundary conditions to be employed in the advective and diffusive
    # solves, respectively
    bcs_a = [dolfin.DirichletBC(Wh[1], T_D_bc, "on_boundary")]
    bcs_d = [dolfin.DirichletBC(Wh[1], T_D_bc, "on_boundary")]

    # Initialise the Lagrangian theta term
    heat_cls.theta_L.assign(1.0)

    # Linear operators underlying the diffusion FE formulation
    A_heat, b_heat = dolfin.PETScMatrix(), dolfin.PETScVector()

    last_step = False
    for j in range(5000):
        step = j+1
        # Update dt, advect particles and project to field
        if use_fixed_dt:
            dt = dt_fixed_vals[i]
        else:
            if abs(float(u0)) > 1e-12:
                dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
            else:
                # Apply some small random noise in the case of pure diffusion
                noise = 0.1 if j % 5 == 0 else 1.0
                dt = noise*dt_fixed_vals[i]

        if t + dt > t_max - 1e-9:
            dt = t_max - t
            last_step = True

        dt_ufl.assign(dt)

        # Step 1: Advect particles
        if abs(float(u0)) > 1e-12:
            ap.do_step(dt)

        # Update time and compute error functionals
        t += dt
        dt_ufl.assign(dt)
        t_ufl.assign(t)
        u_soln.t = t
        u_vec.interpolate(u_soln)

        # Step 2: project advective component
        heat_cls.project_advection([Tstar, Tstarbar], bcs_a)

        if float(kappa) > 1e-12:
            # Step 3: solve for diffusive component
            heat_cls.solve_diffusion(Wh, [T, Tbar], [Tstar, Tstarbar],
                                [A_heat, b_heat], bcs_d,
                                heat_model)
        else:
            # If there is no diffusive component, we just update the
            # temperature solution
            T.vector()[:] = Tstar.vector()

        # Step 4: update particle values
        theta_p = 0.5
        heat_cls.update_field_and_increment_particles(
            [T, Tbar], [Tstar, Tstarbar], theta_p, step, dt)

        if step == 2:
            # We don't immediately have enough information for the linear
            # interpolation of dT/dt between time steps
            # step 1: dTdt0 = 0, dTdt00 = 0 -> theta_L = 1.0
            # step 2: dTdt0 =/= 0, dTdt00 = 0 -> theta_L = 1.0
            # step 3: dTdt0 =/= 0, dTdt00 =/= 0 -> theta_L = 0.5
            heat_cls.theta_L.assign(0.5)

        T_l2 = dolfin.assemble((T_D_bc - T) ** 2 * ufl.dx) ** 0.5
        T_h1 = dolfin.assemble(dolfin.grad(T_D_bc - T) ** 2 * ufl.dx) ** 0.5
        conn = abs(dolfin.assemble(T * ufl.dx) - conservation0)

        update_msg = f"Timestep {j:>4}, Δt={dt:<.3e}, t={t:<.3e}, " \
                     f"‖T - Tₕ‖₂={T_l2:<.3e}, ‖∇T - ∇Tₕ‖₂={T_h1:<.3e}, " \
                     f"Conservation={conn:<.3e}"
        dolfin.info(update_msg)

        if write_output:
            dolfin.XDMFFile("T_debug.xdmf").write_checkpoint(
                T, "T", time_step=t, append=True)
            exact_projector.project(T_D_bc, T_exact)
            dolfin.XDMFFile("T_exact.xdmf").write_checkpoint(
                T_exact, "T_exact", time_step=t, append=True)

            points_list = list(
                dolfin.Point(*pp) for pp in ptcls.positions())
            particles_values_T = ptcls.get_property(1)
            dolfin.XDMFFile(
                os.path.join("particles/T_step%.4d.xdmf" % j)) \
                .write(points_list, particles_values_T)

        if last_step:
            break

    # Final error functionals and write to file
    errors_l2[i] = dolfin.assemble((T_D_bc - T) ** 2 * ufl.dx) ** 0.5
    final_mass_conservation[i] = \
        abs(dolfin.assemble(T * ufl.dx) - conservation0)

# Write final functionals to file
h_rates = np.log(h_vals[:-1] / h_vals[1:])
rates_phi_l2 = np.log(errors_l2[:-1] / errors_l2[1:]) / h_rates
dolfin.info(f"#particles: {n_parts_list}")
dolfin.info(f"mesh h: {h_vals}")
dolfin.info(f"Composition field L2 rates: {rates_phi_l2}")
dolfin.info(f"Composition field l2 error: {errors_l2}")
dolfin.info(f"Conservation change: {final_mass_conservation}")

if IS_TEST_ENV:
    assert np.all(rates_phi_l2 > heat_cls.degree() + 1 - 0.5)
