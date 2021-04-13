import os
import dataclasses
import numpy as np

import ufl
import dolfin
from dolfin import MPI

import dolfin_dg

import leopart

import geopart.stokes
import geopart.stokes.compressible
import geopart.energy.compressible_heat
import geopart.energy.heat
import geopart.projector


assemble_lhs_stokes = False
assemble_lhs_heat = True
solve_heat_as_linear = True

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False
# dolfin.parameters["ghost_mode"] = "shared_facet"


@dataclasses.dataclass
class BenchmarkParameters:
    eta: dolfin.Constant = dolfin.Constant(1.0)
    kappa: dolfin.Constant = dolfin.Constant(1.0)
    Ra: dolfin.Constant = dolfin.Constant(1e4)
    lmbda: dolfin.Constant = dolfin.Constant(1.0)
    ny: int = 64
    nx: int = ny
    t_max: float = 0.25
    c_cfl: float = 1.0

    T_surf: float = 273.0
    DeltaT: float = 3000.0
    gamma_r: dolfin.Constant = dolfin.Constant(1.0)
    alpha: dolfin.Constant = dolfin.Constant(1.0)
    Di: dolfin.Constant = dolfin.Constant(0.25)
    cp: dolfin.Constant = dolfin.Constant(1.0)
    cv: dolfin.Constant = dolfin.Constant(1.0)
    Ks: dolfin.Constant = dolfin.Constant(1.0)
    To: dolfin.Constant = dolfin.Constant(T_surf/DeltaT)


def generate_initial_T(mesh, params):
    T0 = dolfin.Expression(
        "1.0 - x[1] + A*cos(pi*x[0]/lmbda)*sin(pi*x[1])",
        A=0.1, H=1.0, lmbda=params.lmbda, degree=4)
    return T0


class King2010:
    """
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
        Di = params.Di
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
        # from vedo.dolfin import plot
        # plot(mesh)
        # quit()

        # Stokes fields and function spaces
        element_cls = ElementClass()
        W = element_cls.function_space(mesh)
        V = element_cls.velocity_function_space(mesh)
        u_vec = dolfin.Function(V)

        U = element_cls.create_solution_variable(W)

        # Tracers and composition field, space and properties
        x = np.array([[0.0, 0.0]], dtype=np.double)
        s = np.zeros((len(x), 1), dtype=np.float_)
        ptcls = leopart.particles(x, [s, s], mesh)

        heat_prop_idx = 1
        heat_cls = HeatClass(ptcls, u_vec, dt_ufl, heat_prop_idx)
        heat_cls.theta.assign(1.0)
        Sh_vec = heat_cls.function_space()
        T_vec = heat_cls.create_solution_variable(Sh_vec)
        Tstar_vec = heat_cls.create_solution_variable(Sh_vec)

        if isinstance(T_vec, (list, tuple)):
            T, Tbar = T_vec
        else:
            T = T_vec

        # Stokes system model
        p = element_cls.get_pressure(U)
        depth = 1 - ufl.SpatialCoordinate(mesh)[1]
        rhobar = ufl.exp(Di*depth/params.gamma_r)
        Tbar = params.To * ufl.exp(Di*depth)
        f = dolfin.Constant((0, 0))
        # f = rhobar * (Ra * (T - Tbar) - Di * p) * dolfin.Constant((0, 1))
        # f = rhobar * Ra * (T - Tbar) * dolfin.Constant((0, 1))
        model = geopart.stokes.compressible.CompressibleStokesModel(
            eta=eta, f=f, rho=rhobar)
        model.Di = Di
        model.Ra = Ra
        model.T = T
        model.Tbar = Tbar
        u = element_cls.get_velocity(U, model=model)

        # Set the initial system temperature and balance thermal
        # advection by the reaction term
        w = u[1]
        tau = 2*eta*(ufl.sym(ufl.grad(u)) - 1.0/3.0*ufl.div(u)*ufl.Identity(2))
        phi = ufl.inner(tau, ufl.grad(u))
        Q = - Di * rhobar * w * params.alpha * (T + params.To) \
            + phi * Di / Ra
        heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)
        heat_model.rho = rhobar
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
        normal_bcy = dolfin_dg.DGDirichletNormalBC(
            ds(1), dolfin_dg.tangential_proj(u, n))
        normal_bcy.component = 1
        normal_bcx = dolfin_dg.DGDirichletNormalBC(
            ds(2), dolfin_dg.tangential_proj(u, n))
        normal_bcx.component = 0
        weak_bcs = [normal_bcx, normal_bcy]

        # Heat BCs
        ff_heat = dolfin.MeshFunction(
            "size_t", mesh, mesh.topology().dim()-1, 0)
        HEAT_TOP, HEAT_BOTTOM = 2, 1
        dolfin.CompiledSubDomain("near(x[1], 0.0)").mark(ff_heat, HEAT_BOTTOM)
        dolfin.CompiledSubDomain("near(x[1], 1.0)").mark(ff_heat, HEAT_TOP)
        ds_heat = ufl.Measure("ds", subdomain_data=ff_heat, domain=mesh)
        heat_bcs = [
            dolfin_dg.DGDirichletBC(ds_heat(HEAT_TOP), dolfin.Constant(0.0)),
            dolfin_dg.DGDirichletBC(ds_heat(HEAT_BOTTOM), dolfin.Constant(1.0))]

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

            heat_cls.project_advection(Tstar_vec, heat_bcs, model=heat_model)
            heat_cls.solve_diffusion(
                Sh_vec, T_vec, Tstar_vec, heat_mats, heat_bcs, heat_model,
                solve_as_linear_problem=solve_heat_as_linear,
                assemble_lhs=assemble_lhs_heat)

            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=assemble_lhs_stokes)
            projector.project(u, u_vec)
            return u_vec._cpp_object

        ap = AdvectorClass(ptcls, V, uh_accessor, "open")

        # Write particles and their values to XDMF file
        n_particles = MPI.sum(comm, len(ptcls.positions()))

        run_identifier = f"nx{nx}_ny{ny}_Ra{float(Ra)}_CFL{c_cfl}_"\
                         f"nparticles{n_particles}_eta{float(eta):.3f}_" \
                         f"Di{float(Di)}"
        particles_directory = os.path.join(
            directory, run_name, f"particles_{run_identifier}")
        velo_filename = os.path.join(
            directory, run_name, f"velo_{run_identifier}.xdmf")
        pres_filename = os.path.join(
            directory, run_name, f"pres_{run_identifier}.xdmf")
        heat_filename = os.path.join(
            directory, run_name, f"heat_{run_identifier}.xdmf")

        def write_particles_fields(idx, t, append, slice=25):
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
            dolfin.XDMFFile(velo_filename).write_checkpoint(
                u_vec, "u", t, append=append)
            dolfin.XDMFFile(pres_filename).write_checkpoint(
                p, "p", t, append=append)
            dolfin.XDMFFile(heat_filename).write_checkpoint(
                T, "T", t, append=append)

        # Functionals output filename
        dolfin.info(f"Solving with {n_particles} particles")
        data_filename = os.path.join(
            directory, run_name, f"data_{run_identifier}.dat")

        domain_area = dolfin.assemble(dolfin.Constant(1.0)*dx(domain=mesh))
        surface_length = dolfin.assemble(
            dolfin.Constant(1.0)*ds_heat(HEAT_TOP))

        # Compute and output functionals
        def output_data_step(t, dt, append=False):
            urms = (dolfin.assemble(ufl.dot(u_vec, u_vec) * dx) / domain_area
                    ) ** 0.5
            Nu = -dolfin.assemble(ufl.dot(ufl.grad(T), n)*ds_heat(HEAT_TOP)) /\
                dolfin.assemble(T*ds_heat(HEAT_BOTTOM))
            Nu_bot = -dolfin.assemble(
                ufl.dot(ufl.grad(T), n)*ds_heat(HEAT_BOTTOM)) / 1.0
            T_avg = dolfin.assemble(T*dx) / domain_area
            u_top_avg = dolfin.assemble(u[0]*ds_heat(HEAT_TOP)) / surface_length

            conservation = abs(dolfin.assemble(T * dx) - conservation0) \
                           / conservation0
            div_u_l2 = dolfin.assemble(ufl.div(u_vec) ** 2 * dx) ** 0.5

            phi_avg = dolfin.assemble(phi * dx) / domain_area
            phi_avg *= float(Di / Ra)

            W_avg = dolfin.assemble(
                Di * rhobar * T * u[1] * dx) / domain_area

            vals = [t, dt, Nu, Nu_bot, urms, 0.0, u_top_avg, T_avg,
                    phi_avg, W_avg, conservation, div_u_l2]
            if comm.rank == 0:
                with open(data_filename, "a" if append else "w") as fi:
                    msg = ",".join(map("{:.6e}".format, vals)) + "\n"
                    if not append:
                        msg = "# t, dt, Nu_top, Nu_bot, urms, null, " \
                              "u_top_avg, T_avg, phi_avg, W_avg, " \
                              "conservation, \n" + msg
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
        projector = geopart.projector.Projector(u_vec.function_space())
        projector.project(u, u_vec)
        output_data_step(t, 0.0, append=False)
        write_particles_fields(idx=0, t=t, append=False)

        # c_cfl = 10.0
        # Main time loop
        hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())
        last_step = False
        # T_old = dolfin.Function(T.function_space())
        for j in range(50000):
            step = j+1
            # if t > 0.2:
            #     Ra.assign(9e5)
            #     c_cfl = 0.5
            dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
            if not last_step and t + dt > t_max:
                # dt = t_max - t
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

            heat_cls.project_advection(Tstar_vec, heat_bcs, model=heat_model)
            heat_cls.solve_diffusion(
                Sh_vec, T_vec, Tstar_vec, heat_mats, heat_bcs, heat_model,
                solve_as_linear_problem=solve_heat_as_linear,
                assemble_lhs=assemble_lhs_heat)
            heat_cls.update_field_and_increment_particles(
                T_vec, Tstar_vec, 0.5, step, dt
            )

            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=assemble_lhs_stokes)

            # norm_diff = (T.vector() - T_old.vector()).norm("l2") \
            #     / T.vector().norm("l2")
            # if norm_diff < 1e-4:
            #     last_step = True
            dolfin.info(f"Timestep {step:>5d}, dt={dt:>.3e}, t={t:>.3e}, ")
                        # f"T_norm_diff={norm_diff:>.3e}")
            # T.vector().vec().copy(result=T_old.vector().vec())

            # Compute error functionals and write to file
            projector.project(u, u_vec)
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

    cfl_vals = (1.0, 2.0, 4.0, 8.0)
    DiRatF_vals = (
        # (0.25, 1e4, 0.25),
        (1.0, 1e5, 0.15),
    )

    for Di, Ra, tF in DiRatF_vals:
        for c_cfl in cfl_vals:
            params.Di.assign(Di)
            params.Ra.assign(Ra)
            params.c_cfl = c_cfl
            params.t_max = tF
            King2010().run(
                MPI.comm_world,
                geopart.stokes.compressible.HDGConservativeALA,
                geopart.energy.heat.HeatLeastSquaresNonlinearDiffusionHDG1,
                # geopart.energy.heat.HeatPDEConstrainedNonlinearDiffusionHDG2,
                leopart.advect_rk2,
                params,
                output_fields_particles=True
            )

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])