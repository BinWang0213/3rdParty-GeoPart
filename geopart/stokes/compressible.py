import typing

import ufl
import dolfin
import dolfin_dg
import dolfin_dg.hdg_form

import leopart

import geopart.stokes.incompressible
import geopart.timings
import geopart.projector


class CompressibleStokesModel:

    def __init__(self, eta=None, f=None, rho=None):
        self.eta = eta
        self.f = f
        self.rho = rho


class ConservativeFormulation:

    def get_velocity(self, U: dolfin.Function,
                     model: CompressibleStokesModel = None):
        return U.sub(0) / model.rho

    def get_flux_soln(self, U: dolfin.Function,
                      model: CompressibleStokesModel = None):
        return U.sub(0)


class NonConservativeFormulation:

    def get_velocity(self, U: dolfin.Function,
                     model: CompressibleStokesModel = None):
        return U.sub(0)

    def get_flux_soln(self, U: dolfin.Function,
                      model: CompressibleStokesModel = None):
        return U.sub(0)


class TaylorHoodConservative(ConservativeFormulation,
                             geopart.stokes.incompressible.TaylorHood):
    """
    Implementation of the conservative Taylor-Hood element where we solve
    for momentum and pressure, rhou and p, respectively.
    """

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        rhou, p = ufl.split(dolfin.TrialFunction(W))
        v, q = ufl.split(dolfin.TestFunction(W))

        rho = model.rho
        eta = model.eta
        f = model.f

        # u = rhou / rho
        grad_rhou = ufl.grad(rhou)
        grad_u = (grad_rhou*rho - ufl.outer(rhou, ufl.grad(rho)))/rho**2

        # grad_u = ufl.grad(u)
        eye = ufl.Identity(2)

        dx = ufl.dx

        tau = 2 * eta * (ufl.sym(grad_u) - 1.0 / 3.0 * ufl.tr(grad_u) * eye)
        sigma = tau - p * eye
        a = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * dx \
            - q * ufl.div(rhou) * dx
        L = ufl.inner(f, v) * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                L += ufl.dot(bc.get_function(), v) * bc.get_boundary()

        return a, L


class TaylorHoodConservativeALA(ConservativeFormulation,
                             geopart.stokes.incompressible.TaylorHood):
    """
    Implementation of the conservative Taylor-Hood element where we solve
    for momentum and pressure, rhou and p, respectively.
    """

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        rhou, p = ufl.split(dolfin.TrialFunction(W))
        v, q = ufl.split(dolfin.TestFunction(W))

        rho = model.rho
        eta = model.eta
        # f = model.f
        Di = model.Di
        Ra = model.Ra
        T = model.T
        Tbar = model.Tbar

        f = rho * Ra * (T - Tbar) * dolfin.Constant((0, 1))

        # u = rhou / rho
        grad_rhou = ufl.grad(rhou)
        grad_u = (grad_rhou*rho - ufl.outer(rhou, ufl.grad(rho)))/rho**2

        # grad_u = ufl.grad(u)
        eye = ufl.Identity(2)

        dx = ufl.dx

        tau = 2 * eta * (ufl.sym(grad_u) - 1.0 / 3.0 * ufl.tr(grad_u) * eye)
        sigma = tau - p * eye
        a = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * dx \
            - q * ufl.div(rhou) * dx
        a += Di * p * ufl.dot(dolfin.Constant((0, 1)), v) * dx
        L = ufl.inner(f, v) * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                L += ufl.dot(bc.get_function(), v) * bc.get_boundary()

        return a, L


class TaylorHood(NonConservativeFormulation,
                 geopart.stokes.incompressible.TaylorHood):
    """
    Implementation of the standard Taylor-Hood element where we solve for
    velocity and pressure, u and p, respectively
    """

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        u, p = ufl.split(dolfin.TrialFunction(W))
        v, q = ufl.split(dolfin.TestFunction(W))

        rho = model.rho
        eta = model.eta
        f = model.f

        grad_u = ufl.grad(u)
        eye = ufl.Identity(2)

        dx = ufl.dx

        tau = 2 * eta * (ufl.sym(grad_u) - 1.0 / 3.0 * ufl.tr(grad_u) * eye)
        sigma = tau - p * eye
        a = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * dx \
            - q * ufl.div(rho*u) * dx
        L = ufl.inner(f, v) * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                L += ufl.dot(bc.get_function(), v) * bc.get_boundary()

        return a, L


class HDGConservative(ConservativeFormulation,
                      geopart.stokes.StokesElement):
    """
    Implementation of the hybrid discontinuous Galerkin scheme.
    """
    def degree(self) -> int:
        return 1

    def __init__(self):
        self.strong_bcs = None
        self.assembler = None

        solver = dolfin.PETScKrylovSolver()
        solver.set_options_prefix("hdg_stokes_")
        dolfin.PETScOptions.set("hdg_stokes_ksp_type", "preonly")
        dolfin.PETScOptions.set("hdg_stokes_pc_type", "lu")
        dolfin.PETScOptions.set("hdg_stokes_pc_factor_mat_solver_type",
                                "mumps")
        solver.set_from_options()

        self.solver = solver
        self.cfl_projector = None
        self.speed_function = None

    def compute_cfl_dt(self, u_vec, hmin, c_cfl):
        if self.cfl_projector is None:
            speed_V = dolfin.FunctionSpace(
                u_vec.function_space().mesh(), "CG", 1)
            cfl_projector = geopart.projector.Projector(speed_V)
            self.speed_function = dolfin.Function(speed_V)
        cfl_projector.project(ufl.sqrt(ufl.dot(u_vec, u_vec)),
                              self.speed_function)
        max_u_vec = self.speed_function.vector().norm("linf")
        return c_cfl * hmin / max_u_vec

    def stokes_elements(self, mesh: dolfin.Mesh):
        k = self.degree()
        We = ufl.VectorElement("DG", mesh.ufl_cell(), k)
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), k - 1)

        Wbare = ufl.VectorElement("CG", mesh.ufl_cell(), k)["facet"]
        Qbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), k)
        return We, Qe, Wbare, Qbare

    def function_space(self, mesh: dolfin.Mesh):
        We, Qe, Wbare, Qbare = self.stokes_elements(mesh)
        mixed_local = dolfin.FunctionSpace(mesh, ufl.MixedElement([We, Qe]))
        mixed_global = dolfin.FunctionSpace(
            mesh, ufl.MixedElement([Wbare, Qbare]))
        return mixed_local, mixed_global

    def velocity_sub_space(self, W: dolfin.FunctionSpace):
        return W[0].sub(0)

    def velocity_function_space(self, mesh):
        return dolfin.FunctionSpace(mesh, self.stokes_elements(mesh)[0])

    def pressure_function_space(self, mesh):
        return dolfin.FunctionSpace(mesh, self.stokes_elements(mesh)[1])

    def get_velocity(self, U: dolfin.Function,
                     model: CompressibleStokesModel = None):
        return U[0].sub(0) / model.rho

    def get_pressure(self, U: dolfin.Function):
        return U[0].sub(1)

    def create_solution_variable(self, W: dolfin.FunctionSpace):
        return dolfin.Function(W[0]), dolfin.Function(W[1])

    def get_flux_soln(self, U: dolfin.Function,
                      model: CompressibleStokesModel = None):
        return U[0].sub(0)

    @geopart.timings.apply_dolfin_timer
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        k = self.degree()

        W_, Wbar_ = W
        U_, Ubar_ = U

        V_, Vbar_ = dolfin.TestFunction(W_), dolfin.TestFunction(Wbar_)
        v, q = ufl.split(V_)
        vbar, qbar = ufl.split(Vbar_)

        alpha = dolfin.Constant(6 * k ** 2)
        rho = model.rho

        rhou, p = ufl.split(U[0])
        rhoubar, pbar = ufl.split(U[1])
        eta = model.eta
        f = model.f

        dx, dS, ds = ufl.dx, ufl.dS, ufl.ds

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        def F_v(rhou, grad_rhou, p_local=None):
            if p_local is None:
                p_local = pbar
            grad_u = (grad_rhou*rho - ufl.outer(rhou, ufl.grad(rho)))/rho**2
            eye = ufl.Identity(2)
            tau = 2*eta*(ufl.sym(grad_u) - 1.0/3.0*ufl.tr(grad_u) * eye)
            sigma = tau - p_local * eye
            return sigma

        mesh = W_.mesh()
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        penalty = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, rhou)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, rhou, rhoubar, v, vbar, penalty, G, n)

        F = ufl.inner(F_v(rhou, ufl.grad(rhou), p), ufl.grad(v)) * dx
        F += hdg_term.face_residual(dS, ds)
        F += - ufl.inner(f, v) * dx

        # Continuity
        F += ufl.inner(q, ufl.div(rhou)) * dx
        F += facet_integral(ufl.inner(ufl.dot(rhou - rhoubar, n), qbar))

        # Neumann BCs
        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= ufl.dot(bc.get_function(), vbar) * bc.get_boundary()

        # Construct local and global block components
        Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
        J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

        Fr[0] = -Fr[0]
        Fr[1] = -Fr[1]

        def formit(F):
            if isinstance(F, (list, tuple)):
                return list(map(formit, F))
            return dolfin.Form(F)

        Fr = formit(Fr)
        J = formit(J)

        strong_bcs = [dolfin.DirichletBC(Wbar_.sub(0),
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(Wbar_.sub(0).sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in weak_bcs if
                       isinstance(bc, dolfin_dg.DGDirichletNormalBC)]

        self.assembler = leopart.AssemblerStaticCondensation(
            J[0][0], J[0][1],
            J[1][0], J[1][1],
            Fr[0], Fr[1]
        )
        return (Fr, J), strong_bcs

    @geopart.timings.apply_dolfin_timer
    def solve_stokes(self, W, U, mats, weak_bcs, model, assemble_lhs=True):
        if not hasattr(self, "generated_forms"):
            self.generated_forms, self.strong_bcs = self.forms(
                W, U, weak_bcs, model)
            self.strong_bcs_hom = [dolfin.DirichletBC(bc)
                                   for bc in self.strong_bcs]
            for bc in self.strong_bcs_hom:
                bc.homogenize()

        U[0].vector().zero()
        U[1].vector().zero()
        for bc in self.strong_bcs:
            bc.apply(U[0].vector())
            bc.apply(U[1].vector())

        dU = [None]*len(U)
        dU[0] = U[0].copy(deepcopy=True)
        dU[1] = U[1].copy(deepcopy=True)
        # ssc.assemble_global_system(True)
        A, b = mats

        if assemble_lhs:
            self.assembler.assemble_global(A, b)
            self.solver.set_operator(A)
            self.solver.set_reuse_preconditioner(False)
            for bc in self.strong_bcs_hom:
                bc.apply(A, b)
        else:
            self.solver.set_reuse_preconditioner(True)
            self.assembler.assemble_global_rhs(b)
            for bc in self.strong_bcs_hom:
                bc.apply(b)

        dUh, dUhbar = dU
        self.solver.solve(dUhbar.vector(), b)
        self.assembler.backsubstitute(dUhbar._cpp_object, dUh._cpp_object)

        U[0].vector().axpy(1.0, dUh.vector())
        U[1].vector().axpy(1.0, dUhbar.vector())


class HDG2Conservative(HDGConservative):

    def degree(self) -> int:
        return 2


class HDGConservativeALA(HDGConservative):

    @geopart.timings.apply_dolfin_timer
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        k = self.degree()

        W_, Wbar_ = W
        U_, Ubar_ = U

        V_, Vbar_ = dolfin.TestFunction(W_), dolfin.TestFunction(Wbar_)
        v, q = ufl.split(V_)
        vbar, qbar = ufl.split(Vbar_)

        alpha = dolfin.Constant(6 * k ** 2)
        rho = model.rho

        rhou, p = ufl.split(U[0])
        rhoubar, pbar = ufl.split(U[1])
        eta = model.eta
        # f = model.f

        Di = model.Di
        Ra = model.Ra
        T = model.T
        Tbar = model.Tbar

        dx, dS, ds = ufl.dx, ufl.dS, ufl.ds

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        def F_v(rhou, grad_rhou, p_local=None):
            if p_local is None:
                p_local = pbar
            grad_u = (grad_rhou*rho - ufl.outer(rhou, ufl.grad(rho)))/rho**2
            eye = ufl.Identity(2)
            tau = 2*eta*(ufl.sym(grad_u) - 1.0/3.0*ufl.tr(grad_u) * eye)
            sigma = tau - p_local * eye
            return sigma

        mesh = W_.mesh()
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        penalty = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, rhou)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, rhou, rhoubar, v, vbar, penalty, G, n)

        F = ufl.inner(F_v(rhou, ufl.grad(rhou), p), ufl.grad(v)) * dx
        F += hdg_term.face_residual(dS, ds)

        ghat = dolfin.Constant((0.0, -1.0))
        f = rho * (Di * p - Ra * (T - Tbar)) * ghat
        F += - ufl.inner(f, v) * dx

        # Continuity
        F += ufl.inner(q, ufl.div(rhou)) * dx
        F += facet_integral(ufl.inner(ufl.dot(rhou - rhoubar, n), qbar))

        # Neumann BCs
        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= ufl.dot(bc.get_function(), vbar) * bc.get_boundary()

        # Construct local and global block components
        Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
        J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

        Fr[0] = -Fr[0]
        Fr[1] = -Fr[1]

        def formit(F):
            if isinstance(F, (list, tuple)):
                return list(map(formit, F))
            return dolfin.Form(F)

        Fr = formit(Fr)
        J = formit(J)

        strong_bcs = [dolfin.DirichletBC(Wbar_.sub(0),
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(Wbar_.sub(0).sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in weak_bcs if
                       isinstance(bc, dolfin_dg.DGDirichletNormalBC)]

        self.assembler = leopart.AssemblerStaticCondensation(
            J[0][0], J[0][1],
            J[1][0], J[1][1],
            Fr[0], Fr[1]
        )
        return (Fr, J), strong_bcs


class HDG2ConservativeALA(HDGConservativeALA):

    def degree(self) -> int:
        return 2


class HDG(NonConservativeFormulation,
          geopart.stokes.incompressible.HDG):
    """
    Implementation of the hybrid discontinuous Galerkin scheme in
    non-conservative form where we solve for velocity and pressure, u and p,
    respectively.
    """
    def degree(self) -> int:
        return 1

    def get_velocity(self, U: dolfin.Function,
                     model: CompressibleStokesModel = None):
        return U[0].sub(0)

    def get_flux_soln(self, U: dolfin.Function,
                      model: CompressibleStokesModel = None):
        return U[0].sub(0)

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        k = self.degree()

        W_, Wbar_ = W
        U_, Ubar_ = U

        V_, Vbar_ = dolfin.TestFunction(W_), dolfin.TestFunction(Wbar_)
        v, q = ufl.split(V_)
        vbar, qbar = ufl.split(Vbar_)

        alpha = dolfin.Constant(6 * k ** 2)
        rho = model.rho

        u, p = ufl.split(U[0])
        ubar, pbar = ufl.split(U[1])
        eta = model.eta
        f = model.f

        dx, dS, ds = ufl.dx, ufl.dS, ufl.ds

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        def F_v(u, grad_u, p_local=None):
            if p_local is None:
                p_local = pbar
            eye = ufl.Identity(2)
            tau = 2*eta*(ufl.sym(grad_u) - 1.0/3.0*ufl.tr(grad_u) * eye)
            sigma = tau - p_local * eye
            return sigma

        mesh = W_.mesh()
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        penalty = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, u)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, u, ubar, v, vbar, penalty, G, n)

        F = ufl.inner(F_v(u, ufl.grad(u), p), ufl.grad(v)) * dx
        F += hdg_term.face_residual(dS, ds)
        F += - ufl.inner(f, v) * dx

        # Continuity
        F += ufl.inner(q, ufl.div(rho*u)) * dx
        F += facet_integral(ufl.inner(ufl.dot(rho*u - rho*ubar, n), qbar))

        # Neumann BCs
        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= ufl.dot(bc.get_function(), vbar) * bc.get_boundary()

        # Construct local and global block components
        Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
        J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

        Fr[0] = -Fr[0]
        Fr[1] = -Fr[1]

        def formit(F):
            if isinstance(F, (list, tuple)):
                return list(map(formit, F))
            return dolfin.Form(F)

        Fr = formit(Fr)
        J = formit(J)

        strong_bcs = [dolfin.DirichletBC(Wbar_.sub(0),
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(Wbar_.sub(0).sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in weak_bcs if
                       isinstance(bc, dolfin_dg.DGDirichletNormalBC)]

        return (Fr, J), strong_bcs

    def solve_stokes(self, W, U, mats, weak_bcs, model):
        mesh = W[0].mesh()
        if not hasattr(self, "generated_forms"):
            self.generated_forms, self.strong_bcs = self.forms(
                W, U, weak_bcs, model)
            self.strong_bcs_hom = [dolfin.DirichletBC(bc)
                                   for bc in self.strong_bcs]
            for bc in self.strong_bcs_hom:
                bc.homogenize()
        Fr, J = self.generated_forms

        ssc = leopart.StokesStaticCondensation(
            mesh,
            J[0][0], J[0][1],
            J[1][0], J[1][1],
            Fr[0], Fr[1])
        for bc in self.strong_bcs:
            bc.apply(U[0].vector())
            bc.apply(U[1].vector())

        dU = [None]*len(U)
        dU[0] = U[0].copy(deepcopy=True)
        dU[1] = U[1].copy(deepcopy=True)
        ssc.assemble_global_system(True)
        for bc in self.strong_bcs_hom:
            ssc.apply_boundary(bc)

        dUh, dUhbar = dU
        ssc.solve_problem(dUhbar.cpp_object(), dUh.cpp_object(), "mumps",
                          "default")

        U[0].vector().axpy(1.0, dUh.vector())
        U[1].vector().axpy(1.0, dUhbar.vector())


class HDG2(HDG):

    def degree(self) -> int:
        return 2


class HDGDivu(NonConservativeFormulation,
              geopart.stokes.incompressible.HDG):
    """
    Implementation of the hybrid discontinuous Galerkin scheme in
    non-conservative form where we solve for velocity and pressure, u and p,
    respectively.
    """
    def degree(self) -> int:
        return 1

    def get_velocity(self, U: dolfin.Function,
                     model: CompressibleStokesModel = None):
        return U[0].sub(0)

    def get_flux_soln(self, U: dolfin.Function,
                      model: CompressibleStokesModel = None):
        return U[0].sub(0)

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: CompressibleStokesModel):
        k = self.degree()

        W_, Wbar_ = W
        U_, Ubar_ = U

        V_, Vbar_ = dolfin.TestFunction(W_), dolfin.TestFunction(Wbar_)
        v, q = ufl.split(V_)
        vbar, qbar = ufl.split(Vbar_)

        alpha = dolfin.Constant(6 * k ** 2)
        rho = model.rho

        u, p = ufl.split(U[0])
        ubar, pbar = ufl.split(U[1])
        eta = model.eta
        f = model.f

        dx, dS, ds = ufl.dx, ufl.dS, ufl.ds

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        def F_v(u, grad_u, p_local=None):
            if p_local is None:
                p_local = pbar
            eye = ufl.Identity(2)
            tau = 2*eta*(ufl.sym(grad_u) - 1.0/3.0*ufl.tr(grad_u) * eye)
            sigma = tau - p_local * eye
            return sigma

        mesh = W_.mesh()
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        penalty = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, u)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, u, ubar, v, vbar, penalty, G, n)

        F = ufl.inner(F_v(u, ufl.grad(u), p), ufl.grad(v)) * dx
        F += hdg_term.face_residual(dS, ds)
        F += - ufl.inner(f, v) * dx

        # Continuity
        F += ufl.inner(q, ufl.div(u) + ufl.dot(u, ufl.grad(rho))/rho) * dx
        F += facet_integral(ufl.inner(ufl.dot(u - ubar, n), qbar))

        # Neumann BCs
        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= ufl.dot(bc.get_function(), vbar) * bc.get_boundary()

        # Construct local and global block components
        Fr = dolfin_dg.extract_rows(F, [V_, Vbar_])
        J = dolfin_dg.derivative_block(Fr, [U_, Ubar_])

        Fr[0] = -Fr[0]
        Fr[1] = -Fr[1]

        def formit(F):
            if isinstance(F, (list, tuple)):
                return list(map(formit, F))
            return dolfin.Form(F)

        Fr = formit(Fr)
        J = formit(J)

        strong_bcs = [dolfin.DirichletBC(Wbar_.sub(0),
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(Wbar_.sub(0).sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in weak_bcs if
                       isinstance(bc, dolfin_dg.DGDirichletNormalBC)]

        return (Fr, J), strong_bcs

    def solve_stokes(self, W, U, mats, weak_bcs, model):
        mesh = W[0].mesh()
        if not hasattr(self, "generated_forms"):
            self.generated_forms, self.strong_bcs = self.forms(
                W, U, weak_bcs, model)
            self.strong_bcs_hom = [dolfin.DirichletBC(bc)
                                   for bc in self.strong_bcs]
            for bc in self.strong_bcs_hom:
                bc.homogenize()
        Fr, J = self.generated_forms

        ssc = leopart.StokesStaticCondensation(
            mesh,
            J[0][0], J[0][1],
            J[1][0], J[1][1],
            Fr[0], Fr[1])
        for bc in self.strong_bcs:
            bc.apply(U[0].vector())
            bc.apply(U[1].vector())

        dU = [None]*len(U)
        dU[0] = U[0].copy(deepcopy=True)
        dU[1] = U[1].copy(deepcopy=True)
        ssc.assemble_global_system(True)
        for bc in self.strong_bcs_hom:
            ssc.apply_boundary(bc)

        dUh, dUhbar = dU
        ssc.solve_problem(dUhbar.cpp_object(), dUh.cpp_object(), "mumps",
                          "default")

        U[0].vector().axpy(1.0, dUh.vector())
        U[1].vector().axpy(1.0, dUhbar.vector())


class HDGDivu2(HDGDivu):

    def degree(self) -> int:
        return 2
