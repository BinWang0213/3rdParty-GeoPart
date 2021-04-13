import os
import dataclasses
import numpy as np

import ufl
import dolfin
from dolfin import MPI

import dolfin_dg

import leopart

import geopart.stokes
import geopart.stokes.incompressible
import geopart.energy.heat


IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False


@dataclasses.dataclass
class BenchmarkParameters:
    eta: dolfin.Constant = dolfin.Constant(1.0)
    kappa: dolfin.Constant = dolfin.Constant(1.0)
    Ra: dolfin.Constant = dolfin.Constant(21600.0)
    lmbda: dolfin.Constant = dolfin.Constant(1.5)
    ny: int = 60
    nx: int = int(float(lmbda)*ny)
    t_max: float = 5.0
    c_cfl: float = 0.5


def generate_initial_T(mesh, params):
    T0 = dolfin.Expression(
        "1.0 - x[1] + 0.1*cos(pi*x[0]/lmbda)*sin(pi*x[1])",
        H=1.0, lmbda=params.lmbda, degree=4)
    return T0


class BlankenbachCase3:
    """
    Case 3 of the benchmark numerical experiments exhibited in
    Blankenbach et al. 1989. https://doi.org/10.1111/j.1365-246X.1989.tb05511.x
    """

    def run(self, comm, ElementClass, HeatClass, AdvectorClass,
            params: BenchmarkParameters, output_fields_particles=False):

        directory = self.__class__.__name__
        run_name = ElementClass.__name__ \
            + HeatClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
            run_dir = os.path.join(directory, run_name)
            if not os.path.exists(run_dir):
                os.mkdir(run_dir)

        # Benchmark properties
        nx, ny = params.nx, params.ny
        lmbda = float(params.lmbda)
        Ra = params.Ra
        c_cfl = params.c_cfl
        eta = params.eta
        kappa = params.kappa
        dx = dolfin.dx

        dt_ufl = dolfin.Constant(0.0)
        t_ufl = dolfin.Constant(0.0)
        t = 0.0
        t_max = params.t_max

        # Benchmark geometry
        mesh = dolfin.RectangleMesh.create(
            comm, [dolfin.Point(0.0, 0.0), dolfin.Point(float(lmbda), 1.0)],
            [nx, ny], dolfin.CellType.Type.triangle, "left/right")

        # Bias mesh towards core and surface
        bias_degree = 3.0
        coords = np.copy(mesh.coordinates())
        coords[:, 1] = coords[:, 1] - 0.5

        core_vertices = np.where(coords[:, 1] < 0.0)[0]
        surface_vertices = np.where(coords[:, 1] > 0.0)[0]
        coords[core_vertices, 1] = 0.5 * np.tanh(
            bias_degree * (coords[core_vertices, 1]))
        coords[surface_vertices, 1] = 0.5 * np.tanh(
            bias_degree * (coords[surface_vertices, 1]))

        xt = dolfin.MPI.max(mesh.mpi_comm(), np.max(coords[:, 1]))
        r = 0.5/xt
        coords[:, 1] = r*coords[:, 1] + 0.5

        mesh.coordinates()[:] = coords

        # Stokes fields and function spaces
        element_cls = ElementClass()
        W = element_cls.function_space(mesh)
        V = element_cls.velocity_function_space(mesh)
        u_vec = dolfin.Function(V)

        U = element_cls.create_solution_variable(W)
        u = element_cls.get_velocity(U)

        # Tracers and composition field, space and properties
        x = np.array([[0.0, 0.0]], dtype=np.double)
        s = np.zeros((len(x), 1), dtype=np.float_)
        ptcls = leopart.particles(x, [s, s], mesh)

        heat_prop_idx = 1
        heat_cls = HeatClass(ptcls, u_vec, dt_ufl, heat_prop_idx)
        Sh_vec = heat_cls.function_space()
        T_vec = heat_cls.create_solution_variable(Sh_vec)
        Tstar_vec = heat_cls.create_solution_variable(Sh_vec)

        if isinstance(T_vec, (list, tuple)):
            T, Tbar = T_vec
        else:
            T = T_vec

        # Set the initial system temperature and balance thermal
        # advection by the reaction term
        Q = dolfin.Constant(1.0)
        heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)
        T_initial_expr = generate_initial_T(mesh, params)
        heat_cls.T0_a.interpolate(T_initial_expr)
        T.interpolate(T_initial_expr)

        p_degree = heat_cls.degree()
        pmin, pmax = 15*p_degree, 15*p_degree
        if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
            ad = leopart.AddDelete(ptcls, pmin, pmax, [T, heat_cls.dTh0])
            ad.do_sweep()
            ptcls.interpolate(heat_cls.T0_a, heat_prop_idx)
        else:
            leopart.AddDelete(ptcls, pmin, pmax, [T]).do_sweep()

        # Dirichlet boundary conditions in benchmark
        ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        dolfin.CompiledSubDomain(
            "near(x[1], 0.0) or near(x[1], 1.0)").mark(ff, 1)
        dolfin.CompiledSubDomain(
            "near(x[0], 0.0) or near(x[0], lmbda)", lmbda=lmbda).mark(ff, 2)
        ds = ufl.Measure("ds", subdomain_data=ff)

        n = ufl.FacetNormal(mesh)
        normal_bc = dolfin_dg.DGDirichletNormalBC(
            ds(2), dolfin_dg.tangential_proj(u, n))
        normal_bc.component = 0
        weak_bcs = [dolfin_dg.DGDirichletBC(ds(1), dolfin.Constant((0, 0))),
                    normal_bc]

        # Heat BCs
        ff_heat = dolfin.MeshFunction(
            "size_t", mesh, mesh.topology().dim()-1, 0)
        HEAT_TOP, HEAT_BOTTOM = 2, 1
        dolfin.CompiledSubDomain("near(x[1], 0.0)").mark(ff_heat, HEAT_BOTTOM)
        dolfin.CompiledSubDomain("near(x[1], 1.0)").mark(ff_heat, HEAT_TOP)
        ds_heat = ufl.Measure("ds", subdomain_data=ff_heat)
        heat_bcs = [
            dolfin_dg.DGDirichletBC(ds_heat(HEAT_TOP), dolfin.Constant(0.0))]

        # Stokes system model
        f = Ra * T * dolfin.Constant((0, 1))
        model = geopart.stokes.StokesModel(eta=eta, f=f)

        # Particle advector
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

            heat_cls.project_advection(Tstar_vec, [])
            heat_cls.solve_diffusion(
                Sh_vec, T_vec, Tstar_vec, heat_mats, heat_bcs, heat_model)

            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=False)
            velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
            return u_vec._cpp_object

        ap = AdvectorClass(ptcls, V, uh_accessor, "open")

        # Write particles and their values to XDMF file
        n_particles = MPI.sum(comm, len(ptcls.positions()))

        run_identifier = f"nx{nx}_ny{ny}_Ra{float(Ra)}_CFL{c_cfl}_"\
                         f"nparticles{n_particles}_eta{float(eta):.3f}"
        particles_directory = os.path.join(
            directory, run_name, f"particles_{run_identifier}")
        heat_filename = os.path.join(
            directory, run_name, f"heat_{run_identifier}.xdmf")

        def write_particles_fields(idx, t, append, slice=50):
            if not output_fields_particles:
                return
            if idx % slice != 0:
                # The quantity of data output is very large, so we add the
                # option to slice it (output every `slice' iterations)
                return

            if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
                points_list = list(dolfin.Point(*pp)
                                   for pp in ptcls.positions())
                particles_values = ptcls.get_property(heat_prop_idx)
                dolfin.XDMFFile(
                    os.path.join(particles_directory, f"step{idx:0>4d}.xdmf")) \
                    .write(points_list, particles_values)
            dolfin.XDMFFile(heat_filename).write_checkpoint(
                T, "T", t, append=append)

        # Functionals output filename
        dolfin.info(f"Solving with {n_particles} particles")
        data_filename = os.path.join(
            directory, run_name, f"data_{run_identifier}.dat")

        domain_area = dolfin.assemble(dolfin.Constant(1.0)*dx(domain=mesh))

        # Compute and output functionals
        def output_data_step(t, dt, append=False):
            urms = (dolfin.assemble(ufl.dot(u_vec, u_vec) * dx) / domain_area
                    ) ** 0.5
            Nu = -dolfin.assemble(ufl.dot(ufl.grad(T), n)*ds_heat(HEAT_TOP)) /\
                dolfin.assemble(T*ds_heat(HEAT_BOTTOM))
            T_avg = dolfin.assemble(T*dx) / domain_area
            conservation = abs(dolfin.assemble(T * dx) - conservation0)
            div_u_l2 = dolfin.assemble(ufl.div(u_vec) ** 2 * dx) ** 0.5

            vals = [t, dt, urms, Nu, T_avg, conservation, div_u_l2]
            if comm.rank == 0:
                with open(data_filename, "a" if append else "w") as fi:
                    msg = ",".join(map("{:.6e}".format, vals)) + "\n"
                    if not append:
                        msg = "# t, dt, urms, Nu, T_avg, conservation, " \
                              "div_u_l2\n" + msg
                    fi.write(msg)

        # Initial mass and Stokes solve. This problem is isoviscous so the
        # matrix operator need only be assembled in the initial step
        conservation0 = dolfin.assemble(T * dx)
        A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
        element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                 assemble_lhs=True)

        A_heat, b_heat = dolfin.PETScMatrix(), dolfin.PETScVector()
        heat_mats = (A_heat, b_heat)

        # Transfer the computed velocity function and compute functionals
        velocity_assigner = dolfin.FunctionAssigner(
            u_vec.function_space(), element_cls.velocity_sub_space(W))
        velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
        output_data_step(t, 0.0, append=False)
        write_particles_fields(idx=0, t=t, append=False)

        # Main time loop
        hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())
        last_step = False
        for j in range(50000):
            step = j+1

            # Initially we make the time step large to get to steady state
            # solution
            c_cfl = 9.0
            if 1.0 < t < 1.01:
                # After we have reached pseudo-steady state we increase the
                # Rayleigh number to find the harmonic flow
                Ra.assign(216000.0)
                c_cfl = min(params.c_cfl, 1.0)
            elif t >= 1.01 - 1e-6:
                # After a brief stabilisation due to rapid changes in flow,
                # we continue the simulation
                c_cfl = params.c_cfl
            dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
            if not last_step and t + dt > t_max:
                dt = t_max - t
                last_step = True
            dt_ufl.assign(dt)

            if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
                # care: dt_ufl will be reassigned in the RK solver
                ad.do_sweep()
                ap.do_step(dt)

            # Update time and compute the Stokes system solution at the
            # next step
            t += dt
            dt_ufl.assign(dt)
            t_ufl.assign(t)

            heat_cls.project_advection(Tstar_vec, [])
            heat_cls.solve_diffusion(
                Sh_vec, T_vec, Tstar_vec, heat_mats, heat_bcs, heat_model)
            heat_cls.update_field_and_increment_particles(
                T_vec, Tstar_vec, 0.5, step, dt
            )

            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=False)

            dolfin.info(f"Timestep {step:>5d}, dt={dt:>.3e}, t={t:>.3e}")

            # Compute error functionals and write to file
            velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
            output_data_step(t, dt, append=True)
            write_particles_fields(idx=j+1, t=t, append=True)

            if step == 2 and isinstance(
                    heat_cls, geopart.energy.heat.ParticleMethod):
                heat_cls.theta_L.assign(0.5)

            if last_step:
                break


if __name__ == "__main__":
    # Compose and run an experiment
    params = BenchmarkParameters()

    if IS_TEST_ENV:
        # Just make sure it runs without error and set some obscure c_cfl so
        # not to overwrite experimental data
        params.t_max = 2e-2
        params.c_cfl = 1.1351251

    BlankenbachCase3().run(
        MPI.comm_world,
        geopart.stokes.incompressible.HDG,
        geopart.energy.heat.HeatLeastSquaresHDG1,
        leopart.advect_rk2,
        params,
        output_fields_particles=False
    )
