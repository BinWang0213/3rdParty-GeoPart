import dataclasses
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
import geopart.stokes
import geopart.stokes.incompressible
import geopart.timings

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False


@dataclasses.dataclass
class BenchmarkParameters:
    eta_bottom: dolfin.Constant = dolfin.Constant(0.1)
    eta_top: dolfin.Constant = dolfin.Constant(1.0)
    Ra: dolfin.Constant = dolfin.Constant(3e5)
    Rb: dolfin.Constant = dolfin.Constant(4.5e5)
    lmbda: dolfin.Constant = dolfin.Constant(2.0)
    nx: int = 160
    ny: int = 80
    t_max: float = 0.05
    c_cfl: float = 1.0


def generate_initial_T(mesh, params, degree=1):
    u0_code = "(std::pow(lmbda, 7.0/3.0) / std::pow(1.0 + std::pow(lmbda, 4), 2.0/3.0) * std::pow(Ra / (2.0*sqrt(pi)), 2.0/3.0))"
    v0_code = u0_code
    Q_code = "(2.0*sqrt(lmbda / (pi * " + u0_code + ")))"
    z_code = "((x[1]) >  lmbda - DOLFIN_EPS_LARGE ? lmbda - DOLFIN_EPS_LARGE : (x[1]))"
    x_code = "((x[0] > (lmbda - DOLFIN_EPS_LARGE) ? (lmbda - DOLFIN_EPS_LARGE) : x[0]) < DOLFIN_EPS_LARGE ? DOLFIN_EPS_LARGE : x[0])"
    T_u_code = "0.5 * std::erf((1.0 - " + z_code + ")/2.0 * std::sqrt(" + u0_code + "/" + x_code + "))"
    T_l_code = "1.0 - 0.5 * std::erf(" + z_code + "/2.0*std::sqrt(" + u0_code + " / (lmbda - " + x_code + ")))"
    T_r_code = "0.5 + " + Q_code + "/(2.0*std::sqrt(pi)) * std::sqrt(" + v0_code + " / (" + z_code + " + 1.0)) * std::exp(- (" + x_code + "*" + x_code + " * " + v0_code + ")/(4.0*" + z_code + " + 4.0))"
    T_s_code = "0.5 - " + Q_code + "/(2.0*std::sqrt(pi)) * std::sqrt(" + v0_code + " / (2.0 - " + z_code + ")) * std::exp( - std::pow(lmbda - " + x_code + ", 2)*" + v0_code + "/(8.0 - 4.0*" + z_code + "))"
    T_code = "+".join((T_u_code, T_l_code, T_r_code, T_s_code)) + " - 3.0/2.0"
    T_code = "std::max(" + T_code + ", 0.0)"
    T_code = "std::min(" + T_code + ", 1.0)"
    T_expr = dolfin.Expression(T_code, degree=degree, lmbda=params.lmbda,
                               Ra=params.Ra)

    fspace = dolfin.FunctionSpace(mesh, "CG", degree)
    T_initial = dolfin.interpolate(T_expr, fspace)

    # Because the initial condition is ildefined in the corners, create
    # and apply BCs here.
    strong_heat_bcs = [
        dolfin.DirichletBC(fspace, dolfin.Constant(1.0), "near(x[1], 0.0)"),
        dolfin.DirichletBC(fspace, dolfin.Constant(0.0), "near(x[1], 1.0)")]
    for bc in strong_heat_bcs:
        bc.apply(T_initial.vector())
    return T_initial


class StepFunction(dolfin.UserExpression):

    def __init__(self, mesh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = mesh

    def eval_cell(self, values, x, cell):
        c = dolfin.Cell(self.mesh, cell.index)
        if c.midpoint()[1] > 0.025:
            values[0] = 0.0
        else:
            values[0] = 1.0


class ThermoChemBoussinesq:

    def run(self, comm, ElementClass, HeatClass, CompositionClass,
            AdvectorClass, params):
        directory = self.__class__.__name__
        run_name = ElementClass.__name__ + CompositionClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
            run_dir = os.path.join(directory, run_name)
            if not os.path.exists(run_dir):
                os.mkdir(run_dir)

        # Experiment properties
        c_cfl = params.c_cfl
        nx, ny = params.nx, params.ny
        lmbda = params.lmbda
        Ra, Rb = params.Ra, params.Rb
        t_max = params.t_max

        mesh = dolfin.RectangleMesh.create(
            comm,
            [dolfin.Point(0.0, 0.0), dolfin.Point(float(lmbda), 1.0)],
            [nx, ny], dolfin.CellType.Type.triangle, "left/right")

        # Initialise Stokes and composition data structures
        element_cls = ElementClass()

        dt_ufl = dolfin.Constant(0.0)
        t = 0.0
        t_ufl = dolfin.Constant(0.0)

        # Stokes fields and function spaces
        W = element_cls.function_space(mesh)
        V = element_cls.velocity_function_space(mesh)
        u_vec = dolfin.Function(V)

        U = element_cls.create_solution_variable(W)
        u, p = element_cls.get_velocity(U), element_cls.get_pressure(U)

        # Initialise particles. We need slots for [phi_p, T_p, dTdt_p].
        x = np.array([[0.0, 0.0]], dtype=np.float_)
        s = np.zeros((len(x), 1), dtype=np.float_)
        ptcls = leopart.particles(x, [s, s, s], mesh)

        # Initialise composition problem
        phi_property_idx = 1
        composition_cls = CompositionClass(ptcls, u_vec, dt_ufl,
                                           phi_property_idx,
                                           bounded=(0.0, 1.0))
        Wh = composition_cls.function_space()
        phi = dolfin.interpolate(StepFunction(mesh), Wh)
        composition_cls.update_field(phi)

        # Initialise heat problem
        kappa = 1
        heat_property_idx = 2
        heat_cls = HeatClass(ptcls, u_vec, dt_ufl, heat_property_idx)
        Sh_vec = heat_cls.function_space()
        T_vec = heat_cls.create_solution_variable(Sh_vec)
        Tstar_vec = heat_cls.create_solution_variable(Sh_vec)

        if isinstance(T_vec, (list, tuple)):
            Sh, Shbar = Sh_vec
            T, Tbar = T_vec
        else:
            Sh = Sh_vec
            T = T_vec

        # Set the initial system temperature and balance thermal
        # advection by the reaction term
        Q = dolfin.Constant(0.0)
        heat_model = geopart.energy.heat.HeatModel(kappa=kappa, Q=Q)
        T_initial_expr = generate_initial_T(mesh, params)
        heat_cls.T0_a.interpolate(T_initial_expr)
        T.interpolate(T_initial_expr)

        # Add particles to cells in a sweep. Don't need to bound the
        # sweep as we interpolate phi afterwards.
        if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
            leopart.AddDelete(ptcls, 25, 30, [phi, T, heat_cls.dTh0]).do_sweep()
            ptcls.interpolate(heat_cls.T0_a, 2)
        else:
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

            # Project the advective components
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
        # Dirichlet boundary conditions in benchmark
        ff = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        dolfin.CompiledSubDomain(
            "near(x[1], 0.0) or near(x[1], 1.0)").mark(ff, 1)
        dolfin.CompiledSubDomain(
            "near(x[0], 0.0) or near(x[0], lmbda)", lmbda=lmbda).mark(ff, 2)
        ds = ufl.Measure("ds", subdomain_data=ff)

        n = ufl.FacetNormal(mesh)
        normal_bc1 = dolfin_dg.DGDirichletNormalBC(
            ds(1), dolfin_dg.tangential_proj(u, n))
        normal_bc1.component = 1
        normal_bc2 = dolfin_dg.DGDirichletNormalBC(
            ds(2), dolfin_dg.tangential_proj(u, n))
        normal_bc2.component = 0
        weak_bcs = [normal_bc1, normal_bc2]

        # Heat BCs
        ff_heat = dolfin.MeshFunction("size_t", mesh,
                                      mesh.topology().dim()-1, 0)
        dolfin.CompiledSubDomain("near(x[1], 0.0)").mark(ff_heat, 1)
        dolfin.CompiledSubDomain("near(x[1], 1.0)").mark(ff_heat, 2)
        ds_heat = ufl.Measure("ds", subdomain_data=ff_heat)
        heat_bcs = [dolfin_dg.DGDirichletBC(ds_heat(1), dolfin.Constant(1.0)),
                    dolfin_dg.DGDirichletBC(ds_heat(2), dolfin.Constant(0.0))]

        # Form the Stokes system momentum source and residual
        eta = dolfin.Constant(1.0)
        f = Rb * phi * dolfin.Constant((0, -1))
        f += Ra * T * dolfin.Constant((0, 1))

        # Initial Stokes solve to get u(x, t=0) and p(x, t=0)
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
        divu_l2 = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5

        n_particles = MPI.sum(comm, len(ptcls.positions()))
        dolfin.info("Solving with %d particles" % n_particles)

        # if isinstance(heat_cls, geopart.energy.heat.ParticleMethod):
        #     heat_mats = None
        # else:
        heat_mats = dolfin.PETScMatrix(mesh.mpi_comm()), \
                    dolfin.PETScVector(mesh.mpi_comm())

        dolfin.XDMFFile("thermochem_u.xdmf").write_checkpoint(
            u_vec, "u", time_step=0.0, append=False)
        dolfin.XDMFFile("thermochem_phi.xdmf").write_checkpoint(
            phi, "phi", time_step=0.0, append=False)
        dolfin.XDMFFile("thermochem_T.xdmf").write_checkpoint(
            T, "T", time_step=0.0, append=False)

        # Functionals output filename
        dolfin.info("Solving with %d particles" % n_particles)
        data_name = "data_nx%d_ny%d_Ra%.3e_Rb%.3e_CFL%f_nparticles%d.dat" \
            % (nx, ny, float(Ra), float(Rb), c_cfl, n_particles)
        data_filename = os.path.join(directory, run_name, data_name)

        # Compute and output functionals
        domain_volume = dolfin.assemble(dolfin.Constant(1.0)*dx(domain=mesh))
        def output_data_step(t, dt, append=False):
            urms = (dolfin.assemble(ufl.dot(u_vec, u_vec) * dx) /
                    domain_volume) ** 0.5
            conservation = abs(dolfin.assemble(phi * dx) - conservation0)
            div_u_l2 = dolfin.assemble(ufl.div(u_vec) ** 2 * dx) ** 0.5

            vals = [t, dt, urms, conservation, div_u_l2]
            if comm.rank == 0:
                with open(data_filename, "a" if append else "w") as fi:
                    msg = ",".join(map(lambda v: "%.6e" % v, vals)) + "\n"
                    if not append:
                        msg = "# t, dt, urms, conservation, div_u_l2\n" + msg
                    fi.write(msg)

        output_data_step(t, 0.0, append=False)
        last_step = False
        for j in range(100000):
            # Update dt and advect particles
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
            step = j+1
            heat_cls.project_advection(Tstar_vec, [])
            heat_cls.solve_diffusion(Sh_vec, T_vec, Tstar_vec, heat_mats,
                                heat_bcs, heat_model)
            heat_cls.update_field_and_increment_particles(
                T_vec, Tstar_vec, 0.5, step, dt
            )

            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model,
                                     assemble_lhs=False)

            if j % 10 == 0:
                dolfin.XDMFFile("thermochem_u.xdmf").write_checkpoint(
                    u_vec, "u", time_step=t, append=True)
                dolfin.XDMFFile("thermochem_phi.xdmf").write_checkpoint(
                    phi, "phi", time_step=t, append=True)
                dolfin.XDMFFile("thermochem_T.xdmf").write_checkpoint(
                    T, "T", time_step=t, append=True)

                points_list = list(
                    dolfin.Point(*pp) for pp in ptcls.positions())
                particles_values_phi = ptcls.get_property(phi_property_idx)
                dolfin.XDMFFile(
                    os.path.join("particles", "phi_step%.4d.xdmf" % step)) \
                    .write(points_list, particles_values_phi)
                particles_values_T = ptcls.get_property(heat_property_idx)
                dolfin.XDMFFile(
                    os.path.join("particles", "T_step%.4d.xdmf" % step)) \
                    .write(points_list, particles_values_T)

            # Compute error functionals and write to file
            conservation = abs(dolfin.assemble(phi * dx) - conservation0)
            divu_l2 = dolfin.assemble(ufl.div(u) ** 2 * dx) ** 0.5

            # Write functionals to stdout and file
            update_msg = "Timestep %d, Δt = %.3e, t = %.3e, " \
                         % (j, float(dt), float(t))
            update_msg += "|∫φₕ(x, t) - φₕ(x, 0) dx| = %.3e, " \
                          "‖∇ ⋅ uₕ‖₂ = %.3e" % (conservation, divu_l2)
            dolfin.info(update_msg)
            output_data_step(t, 0.0, append=True)

            # On the second and higher steps we have enough information for
            # higher order terms in the Taylor expansion
            if step == 2:
                heat_cls.theta_L.assign(0.5)

            if last_step:
                break


if __name__ == "__main__":
    # Compose and run a series of experiments
    experiment_classes = (
        ThermoChemBoussinesq,
    )

    params = BenchmarkParameters()
    params.nx = 160
    params.ny = 80
    params.c_cfl = 0.25

    if IS_TEST_ENV:
        # Set cheap problem and obscure c_cfl so not to overwrite
        # experimental data
        params.nx = 80
        params.ny = 40
        params.t_max = 4e-4
        params.c_cfl = 1.513425

    tt = dolfin.Timer("geopart: total")
    for experiment in experiment_classes:
        experiment().run(
            MPI.comm_world,
            geopart.stokes.incompressible.HDG,
            geopart.energy.heat.HeatPDEConstrainedHDG1,
            geopart.composition.incompressible.PDEConstrainedDG0,
            leopart.advect_rk2,
            params)
    tt.stop()

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])