import typing

import dolfin
import dolfin_dg
import dolfin_dg.dolfin.hdg_newton
import dolfin_dg.hdg_form
import dolfin_dg.operators
import leopart
import numpy
import ufl
from dolfin import dx
from ufl import grad

import geopart.energy.heat
import geopart.composition.incompressible
import geopart.elem
import geopart.particleprojection
import geopart.timings


class HeatALAPDEConstrainedHDG1(
        geopart.energy.heat.HeatPDEConstrainedHDG1):

    def degree(self):
        return 1

    def function_space(self):
        return self.V, self.Vbar

    def create_solution_variable(self, W: dolfin.FunctionSpace):
        return dolfin.Function(W[0]), dolfin.Function(W[1])

    @geopart.timings.apply_dolfin_timer
    def generate_diffusion_forms(
            self, W: dolfin.FunctionSpace, U: dolfin.Function,
            Ustar: dolfin.Function,
            weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            model: geopart.energy.heat.HeatModel):
        mesh = W[0].mesh()
        Tstar, _ = Ustar

        Wh = self.function_space()
        funcspace_dict = {'FuncSpace_local': Wh[0],
                          'FuncSpace_bar': Wh[1]}

        alpha = dolfin.Constant(20.0 * self.degree() ** 2)
        theta_d = dolfin.Constant(1.0)
        T0_d, T0_dbar = Ustar
        forms_pde = geopart.energy.heat.FormsDiffusion(
            mesh, funcspace_dict).forms_theta(
            Tstar, T0_d, T0_dbar, self.dt_ufl, model.kappa,
            alpha, model.Q, theta_d)

        strong_bcs = [dolfin.DirichletBC(W[1],
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]

        strong_bcs += [bc for bc in weak_bcs
                       if isinstance(bc, dolfin.DirichletBC)]

        return forms_pde, strong_bcs

    @geopart.timings.apply_dolfin_timer
    def solve_diffusion(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                   Ustar: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        if self.solver is None:
            forms_pde, strong_bcs = self.generate_diffusion_forms(W, U, Ustar, weak_bcs, model)
            self.forms_diff = forms_pde
            self.strong_bcs_d = strong_bcs

            solver = dolfin.PETScKrylovSolver()
            dolfin.PETScOptions.set("ksp_type", "preonly")
            dolfin.PETScOptions.set("pc_type", "lu")
            dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

            solver.set_from_options()

            self.solver = solver

            self.assembler = leopart.AssemblerStaticCondensation(
                self.forms_diff['A_d'], self.forms_diff['G_d'],
                self.forms_diff['G_T_d'], self.forms_diff['B_d'],
                self.forms_diff['F_d'], self.forms_diff['H_d'],
                self.strong_bcs_d
            )
        T, Tbar = U
        A, b = mats
        self.assembler.assemble_global(A, b)
        for bc in self.strong_bcs_d:
            bc.apply(A, b)
        self.solver.set_operator(A)
        self.solver.solve(Tbar.vector(), b)
        self.assembler.backsubstitute(Tbar._cpp_object, T._cpp_object)

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, Ustar: dolfin.Function,
                          weak_bcs) -> None:
        Tstar, Tstarbar = Ustar
        if self.pde_projection is None:
            forms_pde_map = self.forms_pde_map

            psi_star = self.T0_a + self.dt_ufl * (
                    (1 - self.theta_L) * self.dTh00 + self.theta_L * self.dTh0)
            forms_pde = forms_pde_map.forms_theta_linear(
                self.T0_a, self.u_vec, self.dt_ufl, theta_map=self.theta,
                theta_L=self.theta_L, dpsi0=self.dTh0, dpsi00=self.dTh00,
                zeta=dolfin.Constant(self.zeta_val), psi_star=psi_star)

            strong_bcs = [dolfin.DirichletBC(
                Ustar[1].function_space(), bc.get_function(),
                bc.get_boundary().subdomain_data(),
                bc.get_boundary().subdomain_id())
                for bc in weak_bcs
                if isinstance(bc, dolfin_dg.DGDirichletBC)]

            strong_bcs += [bc for bc in weak_bcs
                           if isinstance(bc, dolfin.DirichletBC)]

            pde_projection = leopart.PDEStaticCondensation(
                self.ptcls.mesh(), self.ptcls,
                forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                forms_pde['H_a'],
                forms_pde['B_a'],
                forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                strong_bcs, self.property_idx)
            self.pde_projection = pde_projection

        self.pde_projection.assemble(True, True)
        self.pde_projection.solve_problem(Tstarbar.cpp_object(),
                                          Tstar.cpp_object(),
                                          "mumps", "default")

    @geopart.timings.apply_dolfin_timer
    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        T = U[0]
        Tstar = Ustar[0]

        self.T0_a.vector()[:] = Tstar.vector()
        self.dTh00.vector()[:] = self.dTh0.vector()
        self.dTh0.vector()[:] = \
            (T.vector() - Tstar.vector())/dt

        self.ptcls.increment(self.dTh0, self.prop_idxs, theta_p, step, dt)

    def get_heat(self, U: dolfin.Function):
        return U[0]


class HeatPDEConstrainedHDG2(HeatPDEConstrainedHDG1):

    def degree(self):
        return 2


class HeatPDEConstrainedHDG3(HeatPDEConstrainedHDG1):

    def degree(self):
        return 3


class HeatPDEConstrainedNonlinearDiffusionHDG1(HeatPDEConstrainedHDG1):

    @geopart.timings.apply_dolfin_timer
    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        # Here we only need diffusion and reaction operators. Advection is
        # assumed implicitly to be the material derivative
        mesh = W[0].mesh()
        s, sbar = dolfin.TestFunction(W[0]), dolfin.TestFunction(W[1])
        T, Tbar = U
        Tstar, _ = Ustar

        def F_v(u, grad_u):
            return model.kappa*grad_u

        f = model.Q

        alpha = dolfin.Constant(20.0 * self.degree()**2)
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        sigma = alpha / h
        G = dolfin_dg.homogeneity_tensor(F_v, T)
        hdg_term = dolfin_dg.hdg_form.HDGClassicalSecondOrder(
            F_v, T, Tbar, s, sbar, sigma, G, n)

        dTdt = (T - Tstar) / self.dt_ufl
        F = ufl.inner(F_v(T, grad(T)), grad(s)) * dx - f * s * dx
        F += hdg_term.face_residual(dolfin.dS, dolfin.ds)
        F += dTdt * s * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= bc.get_function() * sbar * bc.get_boundary()

        Fr = dolfin_dg.extract_rows(F, [s, sbar])
        J = dolfin_dg.derivative_block(Fr, [T, Tbar])

        strong_bcs = [dolfin.DirichletBC(W[1],
                                         bc.get_function(),
                                         bc.get_boundary().subdomain_data(),
                                         bc.get_boundary().subdomain_id())
                      for bc in weak_bcs
                      if isinstance(bc, dolfin_dg.DGDirichletBC)]

        strong_bcs += [bc for bc in weak_bcs
                       if isinstance(bc, dolfin.DirichletBC)]

        return (Fr, J), strong_bcs

    def assemble(self, forms: typing.Sequence[ufl.Form],
                 mats: typing.Tuple[dolfin.PETScMatrix, dolfin.PETScVector],
                 bcs: typing.Sequence[dolfin_dg.operators.DGBC]):
        pass

    @geopart.timings.apply_dolfin_timer
    def solve_diffusion(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                   Ustar: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        if self.solver is None:
            (Fr, J), strong_bcs = self.generate_diffusion_forms(W, U, Ustar, weak_bcs, model)
            solver = \
                dolfin_dg.dolfin.hdg_newton.StaticCondensationNewtonSolver(
                    Fr, J, strong_bcs)
            self.solver = solver
        T, Tbar = U
        self.solver.solve(T, Tbar)


class HeatPDEConstrainedNonlinearDiffusionHDG2(
        HeatPDEConstrainedNonlinearDiffusionHDG1):

    def degree(self):
        return 2


class HeatLeastSquaresHDG1(HeatPDEConstrainedHDG1):

    def degree(self):
        return 1

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        super().__init__(ptcls, u_vec, dt_ufl, property_idx, zeta_val)
        Vh = self.function_space()[0]
        self.l2p = leopart.l2projection(ptcls, Vh, property_idx)

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, Ustar: dolfin.Function,
                          bcs) -> None:
        Tstar, Tstarbar = Ustar
        self.l2p.project(Tstar.cpp_object())
