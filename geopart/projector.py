import ufl
import ufl.core.expr
import dolfin


class Projector:

    counter = 1

    def __init__(self, V: dolfin.FunctionSpace, dx: ufl.Measure = ufl.dx):
        """
        Utility class for performing projection operations multiple times.

        Notes
        -----
        The mass matrix will be stored and reused for back substitution each
        time the project method is called. The mass matrix is generated upon
        instantiation of this class. An independent PETSc KSP is generated
        for each instantiation of this class.

        Parameters
        ----------
        V : Function space upon which to project
        dx : Optional volume integral measure for subdomain projection
        """
        id = self.counter
        self.counter += 1

        du, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

        lhs = ufl.inner(du, v) * dx

        mass = dolfin.PETScMatrix()
        vec = dolfin.PETScVector()

        dolfin.assemble(lhs, tensor=mass)

        solver = dolfin.PETScKrylovSolver()
        solver.set_operator(mass)

        prefix = "projector%d_" % id
        solver.set_options_prefix(prefix)
        dolfin.PETScOptions.set(prefix + "ksp_type", "preonly")
        dolfin.PETScOptions.set(prefix + "pc_type", "lu")
        dolfin.PETScOptions.set(prefix + "pc_factor_mat_solver_type", "mumps")
        solver.set_from_options()

        self.V = V
        self.v = v
        self.solver = solver
        self.vec = vec
        self.dx = dx

    def project(self, f: ufl.core.expr.Expr, u: dolfin.Function):
        """
        Evaluate the projection of a given expression onto the function space
        provided at instantiation

        Parameters
        ----------
        f : ufl mathematical expression to project
        u : DOLFIN FE function into which to store the solution

        Returns
        -------
        u
        """
        rhs = ufl.inner(f, self.v) * self.dx

        dolfin.assemble(rhs, tensor=self.vec)
        self.solver.solve(u.vector(), self.vec)
        return u
