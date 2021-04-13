import typing

import ufl
from ufl import sym, div, grad, inner, dot, Identity

import dolfin
from dolfin import dx
import leopart

import dolfin_dg
import dolfin_dg.operators

import geopart.timings
import geopart.stokes


class TaylorHood(geopart.stokes.StokesElement):
    """
    Implementation of the standard Taylor-Hood element.
    """

    def __init__(self):
        solver = dolfin.PETScKrylovSolver()
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "lu")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        solver.set_from_options()

        self.solver = solver

    def degree(self) -> int:
        return 2

    def stokes_elements(self, mesh: dolfin.Mesh):
        p = self.degree()
        Ve = ufl.VectorElement("CG", mesh.ufl_cell(), p)
        Qe = ufl.FiniteElement("CG", mesh.ufl_cell(), p-1)
        return Ve, Qe

    def function_space(self, mesh: dolfin.Mesh):
        return dolfin.FunctionSpace(
            mesh, ufl.MixedElement(self.stokes_elements(mesh)))

    def velocity_sub_space(self, W: dolfin.FunctionSpace):
        return W.sub(0)

    def get_velocity(self, U: dolfin.Function):
        return U.sub(0)

    def get_pressure(self, U: dolfin.Function):
        return U.sub(1)

    def velocity_function_space(self, mesh: dolfin.Mesh):
        return dolfin.FunctionSpace(mesh, self.stokes_elements(mesh)[0])

    def pressure_function_space(self, mesh: dolfin.Mesh):
        return dolfin.FunctionSpace(mesh, self.stokes_elements(mesh)[1])

    @geopart.timings.apply_dolfin_timer
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: geopart.stokes.StokesModel):
        u, p = ufl.split(dolfin.TrialFunction(W))
        v, q = ufl.split(dolfin.TestFunction(W))
        eta = model.eta
        f = model.f
        a = inner(2 * eta * sym(grad(u)), sym(grad(v))) * dx - p * div(
            v) * dx - q * div(u) * dx
        L = inner(f, v) * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                L += dot(bc.get_function(), v) * bc.get_boundary()

        return a, L

    @geopart.timings.apply_dolfin_timer
    def assemble(self,
                 forms: typing.Sequence[typing.Union[dolfin.Form, ufl.Form]],
                 mats: typing.Sequence[typing.Union[
                                           dolfin.Vector, dolfin.Matrix]],
                 bcs: typing.Sequence[dolfin_dg.operators.DGBC]) -> None:
        a, L = forms
        A, b = mats
        V = self.velocity_sub_space(L.arguments()[0].function_space())
        strong_bcs = [dolfin.DirichletBC(V,
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in bcs if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(V.sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in bcs
                       if isinstance(bc, dolfin_dg.DGDirichletNormalBC)]
        self.system_assembler = dolfin.SystemAssembler(a, L, strong_bcs)
        self.system_assembler.assemble(A, b)

    @geopart.timings.apply_dolfin_timer
    def solve_stokes(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                     mats: typing.Sequence[typing.Union[
                                               dolfin.Vector, dolfin.Matrix]],
                     weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                     model: geopart.stokes.StokesModel,
                     assemble_lhs=True) -> None:
        # Initial Stokes solve
        a, L = self.forms(W, U, weak_bcs, model)
        A, b = mats

        U.vector()[:] = 0.0  # Homogenise the problem
        self.assemble((a, L), (A, b), weak_bcs)

        self.solver.set_operator(A)
        self.solver.solve(U.vector(), b)

    def compute_cfl_dt(self, u_vec, hmin, c_cfl):
        max_u_vec = u_vec.vector().norm("linf")
        return c_cfl * hmin / max_u_vec


class Mini(TaylorHood):
    """
    Implementation of the equal order MINI scheme.
    """

    def degree(self) -> int:
        return 1

    def stokes_elements(self, mesh: dolfin.Mesh):
        P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), self.degree())
        B = ufl.FiniteElement(
            "Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
        Ve = ufl.VectorElement(ufl.NodalEnrichedElement(P1, B))
        Qe = P1
        return Ve, Qe


class P2BDG1(TaylorHood):
    """
    Implementation of the conforming enriched velocity (P2B) and nonconforming
    pressure (DG1) scheme.
    """

    def degree(self) -> int:
        return 2

    def stokes_elements(self, mesh: dolfin.Mesh):
        p = self.degree()
        P2 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p)
        B = ufl.FiniteElement(
            "Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
        Ve = ufl.VectorElement(ufl.NodalEnrichedElement(P2, B))
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), p - 1)
        return Ve, Qe


class P2DG0(TaylorHood):
    """
    Implementation of the suboptimal conforming velocity (P2) and nonconforming
    piecewise constant pressure (DG0) scheme.
    """

    def degree(self) -> int:
        return 2

    def stokes_elements(self, mesh: dolfin.Mesh):
        Ve = ufl.VectorElement("Lagrange", mesh.ufl_cell(), self.degree())
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), self.degree() - 2)
        return Ve, Qe


class CR(TaylorHood):
    """
    Implementation of the nonconforming Crouzeix-Raviart scheme.
    """
    def degree(self) -> int:
        return 1

    def stokes_elements(self, mesh: dolfin.Mesh):
        p = self.degree()
        Ve = ufl.VectorElement("Crouzeix-Raviart", mesh.ufl_cell(), p)
        Qe = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(),
                               p - 1)
        return Ve, Qe


class DG(TaylorHood):
    """
    Implementation of the nonconforming Crouzeix-Raviart scheme.
    """

    def degree(self) -> int:
        return 1

    def stokes_elements(self, mesh: dolfin.Mesh):
        p = self.degree()
        Ve = ufl.VectorElement("DG", mesh.ufl_cell(), p)
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), p - 1)
        return Ve, Qe

    @geopart.timings.apply_dolfin_timer
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: geopart.stokes.StokesModel):
        u, p = ufl.split(U)
        v, q = ufl.split(dolfin.TestFunction(W))

        eta = model.eta

        def F_v(u, grad_u):
            return 2 * eta * sym(grad_u) - p * Identity(2)

        self.F_v = F_v

        stokes_op = dolfin_dg.StokesOperator(W.mesh(), W, weak_bcs, F_v)
        F = stokes_op.generate_fem_formulation(u, v, p, q)
        F -= inner(model.f, v) * dx

        stokes_nitsche = dolfin_dg.StokesNitscheBoundary(
            F_v, u, p, v, q, delta=-1)
        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletNormalBC):
                F += stokes_nitsche.slip_nitsche_bc_residual(
                    bc.get_function(),
                    dolfin.Constant((0.0, 0.0)),
                    bc.get_boundary()
                )

        a = ufl.derivative(F, U)
        L = -F

        return a, L


class DG2(DG):

    def degree(self) -> int:
        return 2

class HDG(geopart.stokes.StokesElement):
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

    def stokes_elements(self, mesh: dolfin.Mesh):
        k = self.degree()
        We = ufl.VectorElement("DG", mesh.ufl_cell(), k)
        Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), k - 1)

        Wbare = ufl.VectorElement("CG", mesh.ufl_cell(), k)["facet"]
        # Wbare = ufl.VectorElement("DGT", mesh.ufl_cell(), k)
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

    def get_velocity(self, U: dolfin.Function):
        return U[0].sub(0)

    def get_pressure(self, U: dolfin.Function):
        return U[0].sub(1)

    def create_solution_variable(self, W: dolfin.FunctionSpace):
        return dolfin.Function(W[0]), dolfin.Function(W[1])

    @geopart.timings.apply_dolfin_timer
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: geopart.stokes.StokesModel):
        mesh = W[0].mesh()

        k = self.degree()

        mixedL = W[0]
        mixedG = W[1]

        alpha = dolfin.Constant(6 * k ** 2)
        forms_generator = leopart.FormsStokes(mesh, mixedL, mixedG, alpha)
        ufl_forms = forms_generator.ufl_forms(model.eta, model.f)

        v, q, vbar, qbar = forms_generator.test_functions()

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                ufl_forms['S_S'] += dot(bc.get_function(),
                                        vbar) * bc.get_boundary()

        forms_stokes = forms_generator.fem_forms(ufl_forms['A_S'],
                                                 ufl_forms['G_S'],
                                                 ufl_forms['G_ST'],
                                                 ufl_forms['B_S'],
                                                 ufl_forms['Q_S'],
                                                 ufl_forms['S_S'])

        strong_bcs = [dolfin.DirichletBC(mixedG.sub(0),
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]
        strong_bcs += [dolfin.DirichletBC(mixedG.sub(0).sub(bc.component),
                                          dolfin.Constant(0.0),
                                          bc.get_boundary().subdomain_data(),
                                          bc.get_boundary().subdomain_id())
                       for bc in weak_bcs if
                       isinstance(bc, dolfin_dg.DGDirichletNormalBC)]

        return forms_stokes, strong_bcs

    def compute_cfl_dt(self, u_vec, hmin, c_cfl):
        max_u_vec = u_vec.vector().norm("linf")
        return c_cfl * hmin / max_u_vec

    @geopart.timings.apply_dolfin_timer
    def solve_stokes(self, W, U, mats, weak_bcs, model,
                     assemble_lhs=True):
        Uh, Uhbar = U
        if self.assembler is None:
            forms, strong_bcs = self.forms(W, U, weak_bcs, model)
            assembler = leopart.AssemblerStaticCondensation(
               forms['A_S'], forms['G_S'],
               forms['B_S'],
               forms['Q_S'], forms['S_S']
            )
            self.assembler = assembler
            self.strong_bcs = strong_bcs

        A, b = mats
        if assemble_lhs:
            self.assembler.assemble_global(A, b)
            self.solver.set_operator(A)
            self.solver.set_reuse_preconditioner(False)
            for bc in self.strong_bcs:
                bc.apply(A, b)
        else:
            self.solver.set_reuse_preconditioner(True)
            self.assembler.assemble_global_rhs(b)
            for bc in self.strong_bcs:
                bc.apply(b)

        self.solver.solve(Uhbar.vector(), b)
        self.assembler.backsubstitute(Uhbar._cpp_object, Uh._cpp_object)


class HDG2(HDG):

    def degree(self) -> int:
        return 2
