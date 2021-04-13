import abc
import typing
import ufl
import ufl.core.expr
import dolfin
import dolfin_dg
import leopart

import geopart.particleprojection
import geopart.timings


class FormsPDEMapExtended(leopart.FormsPDEMap):

    def __init__(self, mesh, FuncSpace_dict,
                 beta_map=dolfin.Constant(1e-6), ds=dolfin.ds):
        super().__init__(mesh, FuncSpace_dict, beta_map=beta_map, ds=ds)

    @geopart.timings.apply_dolfin_timer
    def forms_theta_pseudo_compressible_linear(
            self, psih0, uh, dt, theta_map, rho,
            theta_L=dolfin.Constant(1.0),
            dpsi0=dolfin.Constant(0.0),
            dpsi00=dolfin.Constant(0.0),
            h=dolfin.Constant(0.0),
            neumann_idx=99, zeta=dolfin.Constant(0),
            psi_star=None):
        (psi, lamb, psibar) = self._trial_functions()
        (w, tau, wbar) = self._test_functions()

        beta_map = self.beta_map
        n = self.n
        facet_integral = self.facet_integral
        dx = dolfin.dx

        if psi_star is None:
            raise NotImplementedError(f"psi star found to be {psi_star}")
            # psi_star = psih0 + (1 - theta_L) * dpsi00 + theta_L * dpsi0

        # LHS contributions
        gamma = ufl.conditional(ufl.ge(ufl.dot(uh, n), 0), 0, 1)

        # Standard formulation
        N_a = (
            facet_integral(beta_map * ufl.dot(psi, w))
            + zeta * ufl.dot(ufl.grad(psi), ufl.grad(w)) * dx
        )
        G_a = (
            rho * ufl.dot(lamb, w) / dt * dx
            - theta_map * ufl.dot(uh, ufl.grad(lamb)) * w * dx
            + theta_map * (1 - gamma) * ufl.dot(uh, n) * lamb * w * self.ds(
                neumann_idx)
        )
        L_a = -facet_integral(beta_map * ufl.dot(psibar, w))
        H_a = facet_integral(ufl.dot(uh, n) * psibar * tau) \
            - ufl.dot(uh, n) * psibar * tau * self.ds(neumann_idx)
        B_a = facet_integral(beta_map * ufl.dot(psibar, wbar))

        # RHS contributions
        Q_a = ufl.dot(dolfin.Constant(0), w) * dx
        R_a = (
            rho * ufl.dot(psi_star, tau) / dt * dx
            + (1 - theta_map) * ufl.dot(uh, ufl.grad(tau)) * psi_star * dx
            - (1 - theta_map) * (1 - gamma) * ufl.dot(uh, n) * psi_star * tau
            * self.ds(neumann_idx)
            - gamma * ufl.dot(h, tau) * self.ds(neumann_idx)
        )
        S_a = facet_integral(dolfin.Constant(0) * wbar)
        return self._fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)


class LeastSquaresDG0(
        geopart.particleprojection.AbstractParticleProjectionScheme):

    def __init__(self, ptcls: leopart.particles, u: dolfin.Function,
                 dt_ufl: ufl.core.expr.Expr, property_idx: int,
                 **kwargs):
        """
        Parameters
        ----------
        ptcls : LEoPart particles object
        u : Velocity function used to advect the particles (necessary in
            PDE-constrained projection)
        dt_ufl : Time step size
        property_idx : The index of the particle data in the LEoPart
            particles object
        kwargs : Passed to the superclass
        """
        mesh = ptcls.mesh()
        self.Vh = dolfin.FunctionSpace(mesh, self.ufl_element(mesh))
        self.l2p = leopart.l2projection(ptcls, self.Vh, property_idx)
        self.u = u
        self.dt_ufl = dt_ufl
        super().__init__(**kwargs)

    def degree(self) -> int:
        return 0

    def ufl_element(self, mesh: dolfin.Mesh) -> ufl.FiniteElementBase:
        return ufl.FiniteElement("DG", mesh.ufl_cell(), self.degree())

    def function_space(self) -> dolfin.FunctionSpace:
        return self.Vh

    def create_solution_variable(self, W: dolfin.FunctionSpace) \
            -> dolfin.Function:
        return dolfin.Function(W)

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, phi: dolfin.Function) -> None:
        if self.bounded in (None, False):
            self.l2p.project(phi.cpp_object())
        else:
            lb, ub = self.bounded
            self.l2p.project(phi.cpp_object(), lb, ub)

    def solve_diffusion(
            self, W: dolfin.FunctionSpace, U: dolfin.Function,
            Ustar: dolfin.Function,
            mats: typing.Sequence[typing.Union[
                dolfin.Vector, dolfin.Matrix]],
            weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            model: geopart.particleprojection.AbstractCompositionModel) \
            -> None:
        raise NotImplementedError("Pure advection only")

    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        pass

    def update_field(self, U: dolfin.Function) -> None:
        pass


class LeastSquaresDG1(LeastSquaresDG0):

    def degree(self) -> int:
        return 1


class LeastSquaresDG2(LeastSquaresDG0):

    def degree(self) -> int:
        return 2


class LeastSquaresDG3(LeastSquaresDG0):

    def degree(self) -> int:
        return 3


class PDEConstrainedDG0(
        geopart.particleprojection.AbstractParticleProjectionScheme):

    def degree(self):
        return 0

    def __init__(self, ptcls: leopart.particles, u: dolfin.Function,
                 dt_ufl: dolfin.Constant, property_idx: int, **kwargs):
        """
        Parameters
        ----------
        ptcls : LEoPart particles object
        u : Velocity function used to advect the particles (necessary in
            PDE-constrained projection)
        dt_ufl : Time step size
        property_idx : The index of the particle data in the LEoPart
            particles object
        kwargs : Passed to the superclass
        """

        super().__init__(**kwargs)

        self.ptcls = ptcls
        mesh = ptcls.mesh()

        self.zeta_val = 0.0 if self.bounded in (None, False) else 25.0

        # Use pure HDG in advection problem, EDG is unstable
        T_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
        Wbar_e = ufl.FiniteElement("DGT", mesh.ufl_cell(), self.degree())

        self.W = dolfin.FunctionSpace(mesh, self.ufl_element(mesh))
        T = dolfin.FunctionSpace(mesh, T_e)
        Wbar = dolfin.FunctionSpace(mesh, Wbar_e) if self.periodic is None \
            else dolfin.FunctionSpace(mesh, Wbar_e,
                                      constrained_domain=self.periodic)
        # lambda_h = dolfin.Function(T)
        self.phibar_h = dolfin.Function(Wbar)
        self.bcs = [dolfin.DirichletBC(
            Wbar, dolfin.Constant(0.), "on_boundary")] \
            if self.periodic is None else []

        FuncSpace_adv = {'FuncSpace_local': self.W, 'FuncSpace_lambda': T,
                         'FuncSpace_bar': Wbar}
        self.forms_pde_map = FormsPDEMapExtended(ptcls.mesh(), FuncSpace_adv)
        self.u = u
        self.dt_ufl = dt_ufl
        self.property_idx = property_idx
        self.pde_projection = None
        self.phi0 = dolfin.Function(self.W)

    def ufl_element(self, mesh: dolfin.Mesh) -> ufl.FiniteElementBase:
        return ufl.FiniteElement("DG", mesh.ufl_cell(), self.degree())

    def function_space(self) -> dolfin.FunctionSpace:
        return self.W

    def create_solution_variable(self, W: dolfin.FunctionSpace) \
            -> dolfin.Function:
        return dolfin.Function(W)

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, phi: dolfin.Function) -> None:
        if self.pde_projection is None:
            theta = dolfin.Constant(0.5)
            forms_pde_map = self.forms_pde_map
            forms_pde = forms_pde_map.forms_theta_linear(
                self.phi0, self.u, self.dt_ufl, theta,
                zeta=dolfin.Constant(self.zeta_val))

            pde_projection = leopart.PDEStaticCondensation(
                self.ptcls.mesh(), self.ptcls,
                forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                forms_pde['H_a'],
                forms_pde['B_a'],
                forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                self.bcs, self.property_idx)
            self.pde_projection = pde_projection

        self.pde_projection.assemble(True, True)
        self.pde_projection.solve_problem(self.phibar_h.cpp_object(),
                                          phi.cpp_object(),
                                          "mumps", "default")

    def solve_diffusion(
            self, W: dolfin.FunctionSpace, U: dolfin.Function,
            Ustar: dolfin.Function,
            mats: typing.Sequence[typing.Union[
                dolfin.Vector, dolfin.Matrix]],
            weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            model: geopart.particleprojection.AbstractCompositionModel) \
            -> None:
        raise NotImplementedError("Pure advection only")

    def update_field(self, phi):
        self.phi0.vector()[:] = phi.vector()


class PDEConstrainedDG1(PDEConstrainedDG0):

    def degree(self) -> int:
        return 1


class PDEConstrainedDG2(PDEConstrainedDG0):

    def degree(self) -> int:
        return 2


class PDEConstrainedDG3(PDEConstrainedDG0):

    def degree(self) -> int:
        return 3
