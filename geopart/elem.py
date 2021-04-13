import abc
import typing

import dolfin
import dolfin_dg
import ufl


class AbstractGeopartScheme(abc.ABC):

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
