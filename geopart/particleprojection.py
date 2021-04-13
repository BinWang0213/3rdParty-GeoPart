import abc
import typing

import dolfin
import dolfin_dg


class AbstractCompositionModel:
    pass


class AbstractParticleProjectionScheme(abc.ABC):

    def __init__(self,
                 bounded: typing.Union[None, typing.Sequence[float]] = None,
                 periodic: typing.Union[None, dolfin.SubDomain] = None):
        self.bounded = bounded
        self.periodic = periodic

    @abc.abstractmethod
    def degree(self) -> int:
        pass

    @abc.abstractmethod
    def ufl_element(self, mesh: dolfin.Mesh):
        """
        The UFL finite element underlying the scheme.

        Parameters
        ----------
        mesh: Problem mesh
        """
        pass

    @abc.abstractmethod
    def function_space(self):
        """
        Acquire DOLFIN function space(s) required for the complete scheme.

        Parameters
        ----------
        mesh: Problem mesh
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def project_advection(self, Ustar: dolfin.Function, bcs) -> None:
        pass

    @abc.abstractmethod
    def solve_diffusion(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                        Ustar: dolfin.Function,
                        mats: typing.Sequence[typing.Union[
                            dolfin.Vector, dolfin.Matrix]],
                        weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                        model: AbstractCompositionModel) -> None:
        pass

    def update_field_and_increment_particles(
            self, U: dolfin.Function, Ustar: dolfin.Function,
            theta_p: float, step: int, dt: float) -> None:
        raise NotImplementedError("Projection scheme does not support update "
                                  "method")

    def update_field(
            self, U: dolfin.Function) -> None:
        raise NotImplementedError("Projection scheme does not support update "
                                  "method")
