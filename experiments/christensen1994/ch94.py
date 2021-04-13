"""
This experiment is provided as supporting information for the paper

    Jones, Sime & van Keken. Burying Earth’s primitive mantle in the slab
    graveyard.

It attempts reproduction of the numerical experiment exhibited in:

    Christensen & Hofmann (1994). Segregation of subducted oceanic crust in
    the convecting mantle. Journal of Geophysical Research.

Also shown in

    Brandenburg & van Keken (2007). Deep storage of oceanic crust in a
    vigorously convecting mantle. Journal of Geophysical Research.

Its purpose is to show the importance of a pointwise divergence free
approximation of the velocity field when advecting particles in geodyanmics
simulations.
"""

import leopart
import numpy as np
import ufl
from dolfin import (
    Constant, TestFunction, TestFunctions, Function, dx, assemble,
    PETScMatrix, PETScVector,
    PETScKrylovSolver, FunctionSpace, PETScOptions, info, BoundingBoxTree,
    RectangleMesh, Point, Mesh, MPI, NewtonSolver, PETScFactory,
    NonlinearProblem, MeshFunction, CompiledSubDomain, DirichletBC,
    FunctionAssigner, between, parameters, split, Measure, Expression,
    PointSource, assign, cells, XDMFFile
)
from ufl import (
    derivative, div, grad, SpatialCoordinate, MixedElement, VectorElement,
    FiniteElement, as_vector, exp, inner, dot, sym
)

parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 4


class LinearDivFreeVelocityProjection:

    counter = 1

    def __init__(self, u_naive: Function, V: FunctionSpace,
                 eps_val: float = 1e9):
        id = self.counter
        self.counter += 1

        eps = Constant(eps_val)

        v = TestFunction(V)
        w = Function(V)
        u = Function(V)
        F = ufl.inner(u - u_naive, v) * dx
        F_aug = eps*div(u)*div(v) * dx + div(w)*div(v) * dx

        F += F_aug
        J = derivative(F, u)

        mass = PETScMatrix()
        assemble(J, tensor=mass)
        vec = PETScVector()

        solver = PETScKrylovSolver()
        solver.set_operator(mass)

        prefix = "div_projector%d_" % id
        solver.set_options_prefix(prefix)
        PETScOptions.set(prefix + "ksp_type", "preonly")
        PETScOptions.set(prefix + "pc_type", "lu")
        PETScOptions.set(prefix + "pc_factor_mat_solver_type", "superlu_dist")
        solver.set_from_options()

        self.u = u
        self.w = w
        self.solver = solver
        self.vec = vec
        self.eps = eps
        self.du = Function(V)
        self.F = F

    def project(self, max_it=10, abs_tol=1e-7):
        eps_val = float(self.eps)
        u, w = self.u, self.w
        du = self.du
        for j in range(max_it):
            assemble(self.F, tensor=self.vec)

            self.solver.solve(du.vector(), self.vec)
            u.vector().axpy(-1.0, du.vector())
            w.vector().axpy(eps_val, u.vector())

            div_u_error = assemble(div(u)**2 * dx)**0.5
            info("Linear div free projector, iteration %d: ||div(u)||_L2 %.3e"
                 % (j+1, div_u_error))
            if div_u_error < abs_tol:
                return u

        info("WARNING: NonlinearDivFreeProject did not converge")
        return u


class ParticleFilter:

    def __init__(self, p0: Point, p1: Point,
                 tracer_filter: callable):
        self.p0, self.p1 = p0, p1
        self.melt_mesh = RectangleMesh(MPI.comm_self, p0, p1, 1, 1)
        self.melt_bbox = BoundingBoxTree()
        self.melt_bbox.build(self.melt_mesh)
        self.mesh_bbox = None
        self.tracer_filter = tracer_filter

    def apply(self, mesh: Mesh,
              particles: leopart.particles):

        if self.mesh_bbox is None:
            self.mesh_bbox = BoundingBoxTree()
            self.mesh_bbox.build(mesh)

        candidate_cells = self.melt_bbox.compute_entity_collisions(
            self.mesh_bbox)[1]
        candidate_cells = tuple(set(candidate_cells))

        num_properties = particles.num_properties()

        num_particles_filtered = 0
        for c in candidate_cells:
            for pi in range(particles.num_cell_particles(c)):
                particle_props = list(particles.property(c, pi, prop_num)
                                      for prop_num in range(num_properties))
                new_props = self.tracer_filter(particle_props)
                for (prop_idx, prop) in enumerate(new_props):
                    particles.set_property(c, pi, prop_idx, prop)
                num_particles_filtered += 1
        particles.relocate()

        num_particles_filtered = MPI.sum(
            MPI.comm_world, num_particles_filtered)

        gdim = mesh.geometry().dim()
        info("ParticleFilter zone: (%s, %s) filtered %d particles"
             % (str(self.p0.array()[:gdim]), str(self.p1.array()[:gdim]),
                num_particles_filtered))


class MeltZone(ParticleFilter):

    def __init__(self, p0: Point, p1: Point, tracer_filter: callable,
                 num_properties: int = 2):
        self.num_properties_at_init = num_properties
        super().__init__(p0, p1, tracer_filter)

    def apply_melt(self, mesh: Mesh,
                   particles: leopart.particles):
        dt = 1.0

        if self.mesh_bbox is None:
            self.mesh_bbox = BoundingBoxTree()
            self.mesh_bbox.build(mesh)

        candidate_cells = self.melt_bbox.compute_entity_collisions(
            self.mesh_bbox)[1]
        candidate_cells = tuple(set(candidate_cells))

        num_properties = particles.num_properties()

        # The Runge--Kutta schemes add a new property onto the
        # end of the particles property template. Therefore we
        # can only assume and hope the user has set all properties
        # and will not add new ones after initialising an RK advector.
        assumed_rk_advector_used = num_properties > self.num_properties_at_init

        # If an RK advector is used, the midpoint is stored at this index, and
        # the velocity at that midpoint is stored in num_properties - 1
        assumed_rk_idx = num_properties - 2

        num_particles_moved = 0
        for c in candidate_cells:
            for pi in range(particles.num_cell_particles(c)):
                particle_props = list(particles.property(c, pi, prop_num)
                                      for prop_num in range(num_properties))
                px = particles.property(c, pi, 0)  # Location
                valid_tracer, move, new_props = self.tracer_filter(
                    px, particle_props)

                if valid_tracer:
                    # Set new properties
                    if new_props is not None:
                        for (prop_idx, prop) in enumerate(new_props):
                            particles.set_property(c, pi, prop_idx, prop)
                    # And move the particle
                    particles.push_particle(dt, move, c, pi)
                    num_particles_moved += 1
                    if assumed_rk_advector_used:
                        particles.set_property(
                            c, pi, assumed_rk_idx, px + move)
        particles.relocate()

        num_particles_moved = MPI.sum(MPI.comm_world, num_particles_moved)

        gdim = mesh.geometry().dim()
        info("Melt zone: (%s, %s) moved %d particles"
             % (str(self.p0.array()[:gdim]), str(self.p1.array()[:gdim]),
                num_particles_moved))


# Our custom implementation of a Newton solver.
class CustomSolver(NewtonSolver):

    def __init__(self, mesh, name):
        self.name = name
        NewtonSolver.__init__(self, mesh.mpi_comm(), PETScKrylovSolver(),
                              PETScFactory.instance())

    # Specify linear solver parameters for PETSc here
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        self.linear_solver().set_options_prefix(self.name)

        # Default is a direct solve, using mumps
        PETScOptions.set(self.name + "ksp_type", "preonly")
        PETScOptions.set(self.name + "pc_type", "lu")
        PETScOptions.set(self.name + "pc_factor_mat_solver_type", "mumps")

        self.linear_solver().set_from_options()


class CustomProblem(NonlinearProblem):
    def __init__(self, a, L, bcs):
        self.a = a
        self.L = L
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomParticleProblem(CustomProblem):
    def __init__(self, a, L, bcs, ptcls):
        self.ptcls = ptcls
        self.Vy = W.sub(0).sub(1)
        CustomProblem.__init__(self, a, L, bcs)

    def F(self, b, x):
        assemble(self.L, tensor=b)

        positions = self.ptcls.positions()
        z = 1.0 - positions[:, 1]
        s = 0.69315
        beta = s / (1 - np.exp(-s)) * np.exp(-s * z)
        particles_values = np.array(
            self.ptcls.get_property(1), dtype=np.float_)

        new_particles_values = particles_values * beta
        point_value_zip = zip((Point(*pp) for pp in positions),
                              new_particles_values)
        ps = PointSource(self.Vy, list(point_value_zip))
        ps.apply(b)

        for bc in self.bcs:
            bc.apply(b, x)


TOP, BOTTOM = 1, 2
b = Constant(np.log(1.0))
c = Constant(np.log(1.0))
khat = as_vector((0, 1))
Ra = Constant(0.0)
Rb = Constant(3.88e5)
dt = Constant(1e-9)
Q = Constant(2.5)

u0 = 500.0
H, L = 1.0, 4.0
mesh = RectangleMesh(Point(0, 0), Point(L, H), 160, 40, "crossed")
x = SpatialCoordinate(mesh)
z = 1 - x[1]

ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain("near(x[1], 0.0)").mark(ff, BOTTOM)
CompiledSubDomain("near(x[1], 1.0)").mark(ff, TOP)
ds = Measure("ds", subdomain_data=ff)

Ve = VectorElement("CG", mesh.ufl_cell(), 2)
Qe = FiniteElement("CG", mesh.ufl_cell(), 1)

W = FunctionSpace(mesh, MixedElement([Ve, Qe]))

U = Function(W)
Un = Function(W)
Upr = Function(W)
u, p = split(U)
un, pn = split(Un)
upr, ppr = split(Upr)
v, q = TestFunctions(W)

Se = FiniteElement("CG", mesh.ufl_cell(), 2)
S = FunctionSpace(mesh, Se)
T, s = Function(S), TestFunction(S)
Tn = Function(S)
Tpr = Function(S)

# ===================
# Boundary conditions
# u_top = Expression(("(x[0] < 2.0 + cos(omega * t) ? u0 : -u0) + omega / 2.0
# * sin(omega * t)", "0.0"), t=0.0, u0=u0, omega=float(pi)*u0/5.0, degree=4)
u_top = Expression(
    ("-u0 * (2 / (1.0 + exp(-2.0*k*(x[0] - (2.0 + cos(pi*u0/5.0*t))))) - 1.0) "
     "+ pi*u0/10.0 * sin(pi*u0/5.0 * t)",
     "0.0"),
    t=0.0, k=10.0, u0=u0, degree=4)


def generate_bcs(W, S):
    bcs_u = [DirichletBC(W.sub(0), u_top,
                         CompiledSubDomain("near(x[1], H)", H=H)),
             DirichletBC(
                 W.sub(0).sub(0), Constant(0.0),
                 CompiledSubDomain("near(x[0], 0.0) or near(x[0], L)", L=L)),
             DirichletBC(W.sub(0).sub(1), Constant(0.0), "near(x[1], 0.0)"),
             DirichletBC(W.sub(1), Constant(0.0),
                         "near(x[0], 0.0) and near(x[1], 0.0)", "pointwise")]

    bcs_t = [DirichletBC(S, Constant(0.0), "near(x[1], 1.0)"),
             DirichletBC(S, Constant(1.0), "near(x[1], 0.0)")]
    return bcs_u, bcs_t


# =================
# Boussinesq forms
x = SpatialCoordinate(mesh)
z = Constant(1) - x[1]
d = Constant(np.log(6.0))  # CH94 says exp(d) = 6
alpha = d / (1-ufl.exp(-d)) * ufl.exp(-d*z)


def thermal_buoyancy(T, v):
    return Ra * T * alpha * dot(khat, v) * dx


eta_0 = Constant(1.0)


def gen_eta(T):
    return eta_0 * exp(-b * (T - 0.5) + c*(z - 0.5))


def stokes(u, p, v, q, eta):
    return (inner(2 * eta * sym(grad(u)), sym(grad(v))) - p * div(v)
            - q * div(u)) * dx


def heat_steady(u, T, s):
    return (dot(grad(T), grad(s)) + dot(u, grad(T)) * s) * dx


# =====================================================================
# Newton steady state solves to "spin up" the physical constants in the
# thermal problem
Wbous = FunctionSpace(mesh, MixedElement([Ve, Qe, Se]))
Ub = Function(Wbous)
ub, pb, Tb = split(Ub)
vb, qb, sb = TestFunctions(Wbous)
Fboussinesq_steady = stokes(ub, pb, vb, qb, gen_eta(Tb)) \
                     - thermal_buoyancy(Tb, vb) \
                     + heat_steady(ub, Tb, sb) - Q * sb * dx
bcs_b_u, bcs_b_t = generate_bcs(Wbous, Wbous.sub(2))
bcs = bcs_b_u + bcs_b_t

Jbous = derivative(Fboussinesq_steady, Ub)
problem_bous = CustomProblem(Jbous, Fboussinesq_steady, bcs)
solver_bous = CustomSolver(mesh, "bous_")


def apply_solve(const, vals):
    for val in vals:
        info("Setting value: %.3e" % val)
        const.assign(val)
        solver_bous.solve(problem_bous, Ub.vector())
        Nu = -assemble(grad(Tb)[1]*ds(TOP)) / assemble(Tb*ds(BOTTOM))
        urms = (assemble(dot(ub, ub)*dx)
                / assemble(Constant(1.0)*dx(domain=mesh)))**0.5
        Tavg = assemble(Tb*dx) / assemble(Constant(1.0)*dx(domain=mesh))
        info("Nu = %.5e, urms = %.5e, Tavg = %.5e" % (Nu, urms, Tavg))


info("Spinning up b")
apply_solve(b, map(np.log, [10.0, 100.0, 1000.0, 10000.0, 25000.0, 65536.0]))
info("Spinning up c")
apply_solve(c, map(np.log, [5.0, 7.5, 10.0, 50.0, 64.0]))
info("Spinning up Ra")
apply_solve(Ra, [1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5])

# =========
# Particles
pres = 333
xp = leopart.RandomRectangle(
    Point(0.0, 0.0), Point(L, H)).generate([4*pres, pres])
sp = np.zeros((len(xp), 1), np.float_)

domain_volume = assemble(Constant(1.0)*dx(domain=mesh))
tracer_volume = domain_volume / float(len(xp))
C_peridotite = 0.125
particle_weight = float(Rb) * tracer_volume * C_peridotite
property_idx = 1
ptcls = leopart.particles(xp, [sp], mesh)

u_vec = Function(W.sub(0).collapse())
u_vec_assigner = FunctionAssigner(u_vec.function_space(), W.sub(0))
GEO_TOL = tracer_volume
bounded_lims = np.array(
    [0.0 + GEO_TOL, L - GEO_TOL, 0.0 + GEO_TOL, H - GEO_TOL],
    dtype=np.float_)

# RK3 in space only. 2nd order accurate in time.
ad = leopart.advect_rk3(ptcls, u_vec.function_space(), u_vec, "bounded",
                        bounded_lims.flatten())

# =========================
# Predictor corrector forms
eta_pr = gen_eta(Tpr)
Fu_pr = stokes(upr, ppr, v, q, eta_pr) - thermal_buoyancy(Tpr, v)
eta_corr = gen_eta(T)
Fu_corr = stokes(u, p, v, q, eta_corr) - thermal_buoyancy(T, v)

# =================
# Temperature forms

Ft_pr = (Tpr - Tn)*s/dt * dx + heat_steady(u, Tpr, s) - Q * s * dx
Ft_corr = (T - Tn)*s/dt * dx\
          + 0.5*(heat_steady(upr, T, s) + heat_steady(u, Tn, s)) - Q * s * dx

# =========================
# Predictor corrector forms
bcs_u, bcs_t = generate_bcs(W, S)

problem_u_pr = CustomParticleProblem(
    derivative(Fu_pr, Upr), Fu_pr, bcs_u, ptcls)
problem_u_corr = CustomParticleProblem(
    derivative(Fu_corr, U), Fu_corr, bcs_u, ptcls)
problem_t_pr = CustomProblem(derivative(Ft_pr, Tpr), Ft_pr, bcs_t)
problem_t_corr = CustomProblem(derivative(Ft_corr, T), Ft_corr, bcs_t)

solver_u_pr = CustomSolver(mesh, "u_pr_")
solver_u_pr.parameters["absolute_tolerance"] = 1e-7
solver_u_corr = CustomSolver(mesh, "u_corr_")
solver_u_corr.parameters["absolute_tolerance"] = 1e-7
solver_t_pr = CustomSolver(mesh, "t_pr_")
solver_t_pr.parameters["absolute_tolerance"] = 1e-7
solver_t_corr = CustomSolver(mesh, "t_corr_")
solver_t_corr.parameters["absolute_tolerance"] = 1e-7

# ==============================================
# Assign Newton solve to transient initial state
assign(U.sub(0), Ub.sub(0))
assign(U.sub(1), Ub.sub(1))
assign(T, Ub.sub(2))
assign(Upr.sub(0), Ub.sub(0))
assign(Upr.sub(1), Ub.sub(1))
assign(Tpr, Ub.sub(2))

# Upr.assign(U)

Tn.assign(T)
# solver_u_corr.solve(problem_u_corr, U.vector())

C_CFL = 1.0
hmin = MPI.min(mesh.mpi_comm(), mesh.hmin())


def update_dt():
    max_u_vel = U.sub(0, deepcopy=True).vector().norm("linf")
    dt.assign(C_CFL * hmin / max_u_vel)
    info("New dt value: %.3e" % float(dt))


# ==========
# Melt zones
# Setup bcs and meltzone monitors
zM = 0.08
zC = zM/8.0


def create_melt_zone(x0, x1):
    def tracer_classifier(xp, props):
        condition = between(xp[0], (x0, x1)) \
                    and between(xp[1], (H - zM, H - zC))
        move = Point(0.0, H - xp[1] - np.random.random() * zC)
        return condition, move, props if condition else None
    mz = MeltZone(Point(x0, H - zM), Point(x1, H - zC), tracer_classifier)
    return mz


# Initial melt to create melted crust
create_melt_zone(0.0, 4.0).apply_melt(mesh, ptcls)

# Top left and right melt zones
mzs = [create_melt_zone(0.0, 0.08), create_melt_zone(3.92, 4.0)]

# Div free velocity projector
linear_divfree_projector = LinearDivFreeVelocityProjection(
    u_vec, u_vec.function_space())

# ==============
# Main time loop
t = 0.0
t_start_particles = 0.04
props_applied = False
# update_dt()
for j in range(100000):
    t += float(dt)
    u_top.t = t

    Un.assign(U)
    Tn.assign(T)

    info("Time step %d. t = %.3e, dt = %.3e" % (j, t, float(dt)))

    # Update particles
    if t > t_start_particles:
        u_vec_assigner.assign(u_vec, U.sub(0))
        u_div_adj = linear_divfree_projector.project(max_it=10, abs_tol=1e-7)
        u_vec.vector()[:] = u_div_adj.vector()
        ad.do_step(float(dt))

    for mz in mzs:
        mz.apply_melt(mesh, ptcls)

    info("Solving Tpr")
    solver_t_pr.solve(problem_t_pr, Tpr.vector())
    info("Solving upr")
    solver_u_pr.solve(problem_u_pr, Upr.vector())

    info("Solving Tcorr")
    solver_t_corr.solve(problem_t_corr, T.vector())
    info("Solving ucorr")
    solver_u_corr.solve(problem_u_corr, U.vector())

    # Write functionals to stdout
    Nu = -assemble(grad(Tb)[1] * ds(TOP)) / assemble(Tb * ds(BOTTOM))
    urms = (assemble(dot(ub, ub) * dx)
            / assemble(Constant(1.0) * dx(domain=mesh))) ** 0.5
    Tavg = assemble(Tb * dx) / assemble(Constant(1.0) * dx(domain=mesh))
    info("Nu = %.5e, urms = %.5e, Tavg = %.5e" % (Nu, urms, Tavg))

    #
    if j % 40 == 0:
        # plot(T, interactive=False)
        XDMFFile("T.xdmf").write_checkpoint(
            T, "T", time_step=t, append=j != 0)
        XDMFFile("u.xdmf").write_checkpoint(
            U.sub(0), "u", time_step=t, append=j != 0)

        points_list = list(Point(*pp) for pp in ptcls.positions())
        particles_values = ptcls.get_property(property_idx)
        XDMFFile("particles/particles_%.8f.xdmf" % float(t)).write(
            points_list, particles_values)

    if not props_applied and t > t_start_particles:
        particle_weight = Point(particle_weight)
        for cell in cells(mesh):
            c = cell.index()
            for pi in range(ptcls.num_cell_particles(c)):
                ptcls.set_property(c, pi, 1, particle_weight)
        props_applied = True

    update_dt()
