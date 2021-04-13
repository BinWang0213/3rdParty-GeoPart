import abc
import typing

import dolfin
import dolfin_dg
import dolfin_dg.operators
import ufl
from dolfin import dx
from ufl import grad, dot

import geopart.elem


class PoissonModel:

    def __init__(self, kappa=None, Q=None):
        self.kappa = kappa
        self.Q = Q


class PoissonElement(geopart.elem.AbstractGeopartScheme):

    @abc.abstractmethod
    def heat_element(self, mesh: dolfin.Mesh):
        """
        The UFL finite element underlying the scheme.

        Parameters
        ----------
        mesh: Problem mesh
        """
        pass

    @abc.abstractmethod
    def function_space(self, mesh: dolfin.Mesh):
        """
        Generate DOLFIN function space(s) required for the complete scheme.

        Parameters
        ----------
        mesh: Problem mesh
        """
        pass

    def create_solution_variable(self, W: dolfin.FunctionSpace) -> dolfin.Function:
        """
        Given function space(s), generate DOLFIN FE function(s) into which
        the system solution may be stored.

        Parameters
        ----------
        W: Problem finite element space

        Returns
        -------
        An appropriate DOLFIN function(s)
        """
        return dolfin.Function(W)

    @abc.abstractmethod
    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: PoissonModel):
        """
        Generate the UFL FE formulation required by the numerical scheme.

        Parameters
        ----------
        W: Complete FE space(s)
        U: Compute solution function(s)
        weak_bcs: Problem BCs
        model: Viscosity and momentum source model
        """
        pass

    @abc.abstractmethod
    def assemble(self, forms: typing.Sequence[ufl.Form],
                 mats: typing.Tuple[dolfin.PETScMatrix, dolfin.PETScVector],
                 bcs: typing.Sequence[dolfin_dg.operators.DGBC]):
        """
        Assemble the provided forms into the given linear algebra objects.

        Parameters
        ----------
        forms: UFL FE formulations
        mats: Matrices/Vectors required for the linear solution
        bcs: Problem BCs
        """
        pass

    @abc.abstractmethod
    def get_heat(self, U: dolfin.Function):
        """
        Given a complete system solution function(s), extract and return only
        the pressure component.

        Parameters
        ----------
        U: Complete system solution
        """
        return U


class PoissonCG1(PoissonElement):

    def __init__(self):
        solver = dolfin.PETScKrylovSolver()
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "lu")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        solver.set_from_options()

        self.solver = solver

    def degree(self):
        return 1

    def heat_element(self, mesh: dolfin.Mesh):
        return ufl.FiniteElement("CG", mesh.ufl_cell(), self.degree())

    def function_space(self, mesh: dolfin.Mesh):
        return dolfin.FunctionSpace(mesh, self.heat_element(mesh))

    def create_solution_variable(self, W: dolfin.FunctionSpace) -> dolfin.Function:
        return dolfin.Function(W)

    def forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
              weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
              model: PoissonModel):
        u = dolfin.TrialFunction(W)
        v = dolfin.TestFunction(W)
        kappa = model.kappa
        Q = model.Q
        a = dot(kappa*grad(u), grad(v))*dx
        L = Q * v * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                L += bc.get_function() * v * bc.get_boundary()

        return a, L

    def assemble(self, forms: typing.Sequence[ufl.Form],
                 mats: typing.Tuple[dolfin.PETScMatrix, dolfin.PETScVector],
                 bcs: typing.Sequence[dolfin_dg.operators.DGBC]):
        a, L = forms
        A, b = mats
        V = L.arguments()[0].function_space()
        strong_bcs = [dolfin.DirichletBC(V,
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in bcs if isinstance(bc, dolfin_dg.DGDirichletBC)]
        self.system_assembler = dolfin.SystemAssembler(a, L, strong_bcs)
        self.system_assembler.assemble(A, b)

    def solve_heat(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: PoissonModel) -> None:
        # Initial Stokes solve
        a, L = self.forms(W, U, weak_bcs, model)
        A, b = mats

        U.vector()[:] = 0.0  # Homogenise the problem
        self.assemble((a, L), (A, b), weak_bcs)

        self.solver.set_operator(A)
        self.solver.solve(U.vector(), b)

    def get_heat(self, U: dolfin.Function):
        return U


class PoissonCG2(PoissonCG1):

    def degree(self):
        return 2
