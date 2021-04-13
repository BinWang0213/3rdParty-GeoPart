import dataclasses
import os

import dolfin
import dolfin_dg
import leopart
import numpy as np
import ufl
from dolfin import MPI

import geopart.composition.incompressible
import geopart.stokes
import geopart.stokes.incompressible

IS_TEST_ENV = "PYTEST_CURRENT_TEST" in os.environ
dolfin.parameters["std_out_all_processes"] = False


@dataclasses.dataclass
class BenchmarkParameters:
    eta_bottom: dolfin.Constant = dolfin.Constant(0.1)
    eta_top: dolfin.Constant = dolfin.Constant(1.0)
    Rb: dolfin.Constant = dolfin.Constant(1.0)
    lmbda: dolfin.Constant = dolfin.Constant(0.9142)
    nx: int = 40
    ny: int = 40
    db: float = 0.2
    t_max: float = 200.0
    c_cfl: float = 1.0


class StepFunction(dolfin.UserExpression):
    """
    Expression used to define the initial composition field configuration. The
    initial compositionally dense and light layers lie above and below y = db,
    respectively. This initial state is slightly perturbed from equilibrium.
    """

    def __init__(self, mesh, db, lmbda, **kwargs):
        self.mesh = mesh
        self.db = float(db)
        self.lmbda = float(lmbda)
        super().__init__(self, **kwargs)

    def eval_cell(self, values, x, cell):
        c = dolfin.Cell(self.mesh, cell.index)
        y = c.midpoint()[1]
        if y > self.db + 0.02 * np.cos(np.pi * x[0] / self.lmbda):
            values[0] = 1.0
        else:
            values[0] = 0.0


class RayleighTaylor:
    """
    Rayleigh-Taylor instability benchmark problem in geodynamics as
    documented in https://doi.org/10.1029/97JB01353
    """

    def run(self, comm, ElementClass, CompositionClass, AdvectorClass,
            params: BenchmarkParameters, output_fields_particles=False):

        directory = self.__class__.__name__
        run_name = ElementClass.__name__ \
            + CompositionClass.__name__ \
            + AdvectorClass.__name__

        if comm.rank == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
            run_dir = os.path.join(directory, run_name)
            if not os.path.exists(run_dir):
                os.mkdir(run_dir)

        # Benchmark properties
        db = params.db
        nx, ny = params.nx, params.ny
        lmbda = params.lmbda
        Rb = params.Rb
        c_cfl = params.c_cfl

        dt_ufl = dolfin.Constant(0.0)
        t_ufl = dolfin.Constant(0.0)
        t = 0.0
        t_max = params.t_max

        # Benchmark geometry
        mesh = dolfin.RectangleMesh.create(
            comm, [dolfin.Point(0.0, 0.0), dolfin.Point(float(lmbda), 1.0)],
            [nx, ny], dolfin.CellType.Type.triangle, "left/right")

        # Shift the mesh to line up with the initial step function condition
        scale = db * (1.0 - db)
        shift = dolfin.Expression(
            ("0.0", "x[1]*(H - x[1])/S*A*cos(pi/L*x[0])"),
            A=0.02, L=lmbda, H=1.0, S=scale, degree=4)

        V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
        displacement = dolfin.interpolate(shift, V)
        dolfin.ALE.move(mesh, displacement)

        # Entrainment functional measures
        de = 1
        cf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        dolfin.CompiledSubDomain("x[1] > db - DOLFIN_EPS", db=db).mark(cf, de)
        dx = ufl.Measure("dx", subdomain_data=cf)

        # Stokes fields and function spaces
        element_cls = ElementClass()
        W = element_cls.function_space(mesh)
        V = element_cls.velocity_function_space(mesh)
        u_vec = dolfin.Function(V)

        U = element_cls.create_solution_variable(W)
        u, p = element_cls.get_velocity(U), element_cls.get_pressure(U)

        # Tracers and composition field, space and properties
        x = np.array([[0.0, 0.0]], dtype=np.double)
        s = np.zeros((len(x), 1), dtype=np.float_)
        ptcls = leopart.particles(x, [s], mesh)

        composition_cls = CompositionClass(ptcls, u_vec, dt_ufl, 1,
                                           bounded=(0.0, 1.0))
        Wh = composition_cls.function_space()
        phi = dolfin.interpolate(StepFunction(mesh, db, lmbda), Wh)
        composition_cls.update_field(phi)

        ad = leopart.AddDelete(ptcls, 25, 30, [phi], [1], (0.0, 1.0))
        ad.do_sweep()

        property_idx = 1
        ptcls.interpolate(phi, property_idx)

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

        # Stokes system model
        f = Rb * phi * dolfin.Constant((0, -1))
        eta_top = params.eta_top
        eta_bottom = params.eta_bottom
        eta = eta_bottom + phi * (eta_top - eta_bottom)
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
            composition_cls.project_advection(phi)
            phi.vector().update_ghost_values()
            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)
            velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
            return u_vec._cpp_object

        ap = AdvectorClass(ptcls, V, uh_accessor, "open")

        # Write particles and their values to XDMF file
        n_particles = MPI.sum(comm, len(ptcls.positions()))

        particles_directory = os.path.join(
            directory, run_name,
            "particles_nx%d_ny%d_Rb%f_CFL%f_nparticles%d_etabot%.3f"
            % (nx, ny, float(Rb), c_cfl, n_particles, float(eta_bottom)))
        composition_filename = os.path.join(
            directory, run_name,
            "composition_nx%d_ny%d_Rb%f_CFL%f_nparticles%d_etabot%.3f.xdmf"
            % (nx, ny, float(Rb), c_cfl, n_particles, float(eta_bottom)))

        def write_particles_fields(idx, t, append):
            if not output_fields_particles:
                return
            points_list = list(dolfin.Point(*pp) for pp in ptcls.positions())
            particles_values = ptcls.get_property(property_idx)
            dolfin.XDMFFile(
                os.path.join(particles_directory, "step%.4d.xdmf" % idx)) \
                .write(points_list, particles_values)
            dolfin.XDMFFile(composition_filename).write_checkpoint(
                phi, "composition", t, append=append)

        # Functionals output filename
        dolfin.info("Solving with %d particles" % n_particles)
        data_name = "data_nx%d_ny%d_Rb%f_CFL%f_nparticles%d_etabot%.3f.dat" \
            % (nx, ny, float(Rb), c_cfl, n_particles, float(eta_bottom))
        data_filename = os.path.join(directory, run_name, data_name)

        # Compute and output functionals
        def output_data_step(t, dt, append=False):
            urms = (1.0 / lmbda * dolfin.assemble(ufl.dot(u_vec, u_vec) *
                                                  dx)) ** 0.5
            conservation = abs(dolfin.assemble(phi * dx) - conservation0)
            entrainment = dolfin.assemble(1.0 / (lmbda * dolfin.Constant(db)) *
                                          phi * dx(de))
            div_u_l2 = dolfin.assemble(ufl.div(u_vec) ** 2 * dx) ** 0.5

            vals = [t, dt, urms, conservation, entrainment, div_u_l2]
            if comm.rank == 0:
                with open(data_filename, "a" if append else "w") as fi:
                    msg = ",".join(map(lambda v: "%.6e" % v, vals)) + "\n"
                    if not append:
                        msg = "# t, dt, urms, conservation, entrainment, " \
                              "div_u_l2\n" + msg
                    fi.write(msg)

        # Initial mass and Stokes solve
        conservation0 = dolfin.assemble(phi * dx)
        A, b = dolfin.PETScMatrix(), dolfin.PETScVector()
        element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

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
            dt = element_cls.compute_cfl_dt(u_vec, hmin, c_cfl)
            if not last_step and t + dt > t_max:
                dt = t_max - t
                last_step = True
            dt_ufl.assign(dt)

            # dt_ufl will be reassigned in the RK solver
            ap.do_step(dt)
            dt_ufl.assign(dt)

            ad.do_sweep_failsafe(4)  # Robust for non div free methods
            composition_cls.project_advection(phi)
            phi.vector().update_ghost_values()

            # Update time and compute the Stokes system solution at the
            # next step
            t += dt
            t_ufl.assign(t)
            element_cls.solve_stokes(W, U, (A, b), weak_bcs, model)

            dolfin.info("Timestep %d, dt = %.3e, t = %.3e" % (j, dt, t))

            # Compute error functionals and write to file
            velocity_assigner.assign(u_vec, element_cls.get_velocity(U))
            output_data_step(t, dt, append=True)
            write_particles_fields(idx=j+1, t=t, append=True)

            composition_cls.update_field(phi)

            if last_step:
                break


if __name__ == "__main__":
    # Compose and run an experiment
    params = BenchmarkParameters()

    if IS_TEST_ENV:
        params.t_max = 30.0
        params.c_cfl = 1.14252

    RayleighTaylor().run(
        MPI.comm_world,
        geopart.stokes.incompressible.HDG,
        geopart.composition.incompressible.PDEConstrainedDG1,
        leopart.advect_rk2,
        params,
        output_fields_particles=False
    )
