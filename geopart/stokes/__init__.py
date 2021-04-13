import abc
import typing

import dolfin
import dolfin_dg.operators
import ufl


class StokesModel:

    def __init__(self, eta=None, f=None):
        self.eta = eta
        self.f = f


class StokesElement(abc.ABC):

    @abc.abstractmethod
    def degree(self) -> int:
        pass

    @abc.abstractmethod
    def stokes_elements(self, mesh: dolfin.Mesh):
        """
        The UFL finite elements underlying the scheme.

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

    @abc.abstractmethod
    def velocity_function_space(self, mesh: dolfin.Mesh):
        """
        Generate a function space solely for definition of the velocity
        function.

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
              model):
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
    def compute_cfl_dt(self, u_vec: dolfin.Function, hmin: float,
                       c_cfl: float):
        """
        Given the velocity function, compte an appropriate estimate for optimal
        time step size based on the CFL criterion.

        Parameters
        ----------
        u_vec: Velocity function
        hmin: Minimum spacial characteristic (e.g. smallest mesh cell size)
        c_cfl: CFL number
        """
        pass

    @abc.abstractmethod
    def get_velocity(self, U: dolfin.Function):
        """
        Given a complete system solution function(s), extract and return only
        the velocity component.

        Parameters
        ----------
        U: Complete system solution
        """
        pass

    @abc.abstractmethod
    def get_pressure(self, U: dolfin.Function):
        """
        Given a complete system solution function(s), extract and return only
        the pressure component.

        Parameters
        ----------
        U: Complete system solution
        """
        pass