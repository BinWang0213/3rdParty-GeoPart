import abc
import typing

import numpy

import dolfin
import dolfin_dg
import dolfin_dg.operators
import dolfin_dg.hdg_form
import dolfin_dg.dolfin.hdg_newton
import ufl
from dolfin import dx
from ufl import grad, dot

import leopart

import geopart.particleprojection
import geopart.elem
import geopart.composition.incompressible
import geopart.timings


class FormsDiffusion:
    """
    Initializes the forms for the diffusion problem
    following Labeur and Wells (2007)

    Minor modification of the code originally written by J. Maljaars 2020.
    """

    def __init__(self, mesh, FuncSpace_dict):
        self.W = FuncSpace_dict['FuncSpace_local']
        self.Wbar = FuncSpace_dict['FuncSpace_bar']

        self.n = ufl.FacetNormal(mesh)
        self.he = ufl.CellDiameter(mesh)

    def forms_theta(self, phih_star, phih0, phibarh0d, dt,
                    kappa, alpha, f, theta_d):
        phi, w = ufl.TrialFunction(self.W), ufl.TestFunction(self.W)
        phibar = ufl.TrialFunction(self.Wbar)
        wbar = ufl.TestFunction(self.Wbar)

        facet_integral = self.facet_integral
        he = self.he
        n = self.n
        beta_d = -alpha * kappa / he

        # LHS contributions
        A_d = 1. / dt * phi * w * dx \
            + theta_d * dot(kappa * grad(phi), grad(w)) * dx \
            - theta_d * facet_integral(kappa * dot(grad(phi), n) * w
                                       + kappa * dot(phi * n, grad(w))) \
            - theta_d * facet_integral(beta_d * phi * w)
        G_d = theta_d * facet_integral(beta_d * phibar * w) \
            + theta_d * facet_integral(kappa * dot(phibar * n, grad(w)))
        G_T_d = -theta_d * facet_integral(beta_d * phi * wbar) \
                - theta_d * facet_integral(kappa * dot(wbar * n, grad(phi)))
        B_d = theta_d * facet_integral(beta_d * phibar * wbar)

        # RHS contributions
        F_d = 1. / dt * phih_star * w * dx \
            - (1 - theta_d) * dot(kappa * grad(phih0), grad(w)) * dx \
            + (1 - theta_d) * facet_integral(
                kappa * dot(grad(phih0), n) * w
                + kappa * dot(phih0 * n, grad(w))) \
            + (1 - theta_d) * facet_integral(beta_d * phih0 * w) \
            - (1 - theta_d) * facet_integral(beta_d * phibarh0d * w) \
            - (1 - theta_d) * facet_integral(
                kappa * dot(phibarh0d * n, grad(w))) \
            + f * w * dx
        H_d = facet_integral(dolfin.Constant(0) * wbar) \
            + (1 - theta_d) * facet_integral(beta_d * phih0 * wbar) \
            + (1 - theta_d) * facet_integral(
            kappa * dot(wbar * n, grad(phih0))) \
            - (1 - theta_d) * facet_integral(beta_d * phibarh0d * wbar)
        return self.__get_form_dict(A_d, G_d, G_T_d, B_d, F_d, H_d)

    def facet_integral(self, integrand):
        return integrand('-') * ufl.dS + integrand('+') * ufl.dS \
               + integrand * ufl.ds

    def __get_form_dict(self, A_d, G_d, G_T_d, B_d, F_d, H_d):
        # Turn into forms
        A_d = dolfin.Form(A_d)
        G_d = dolfin.Form(G_d)
        G_T_d = dolfin.Form(G_T_d)
        B_d = dolfin.Form(B_d)
        F_d = dolfin.Form(F_d)
        H_d = dolfin.Form(H_d)

        # Return dict of forms
        return {'A_d': A_d, 'G_d': G_d, 'G_T_d': G_T_d, 'B_d': B_d,
                'F_d': F_d, 'H_d': H_d}


class HeatModel(geopart.particleprojection.AbstractCompositionModel):

    def __init__(self, kappa=None, Q=None, rho=None):
        self.kappa = kappa
        self.Q = Q
        self.rho = rho


class ParticleMethod:
    pass


class HeatPDEConstrainedHDG1(
        geopart.particleprojection.AbstractParticleProjectionScheme,
        ParticleMethod):

    def degree(self):
        return 1

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        self.dt_ufl = dt_ufl
        self.u_vec = u_vec
        self.property_idx = property_idx
        self.pde_projection = None
        self.zeta_val = zeta_val
        self.ptcls = ptcls

        self.theta = dolfin.Constant(1.0)  # Backward Euler & 2nd order method
        self.theta_L = dolfin.Constant(1.0)  # TODO: Must change to 0.5 on 2nd
        # step

        self.periodic = None  # TODO: Account for BCs

        mesh = ptcls.mesh()
        T_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
        Ve, Vbare = self.ufl_element(mesh)

        self.V = dolfin.FunctionSpace(mesh, Ve)
        self.langrange_space = dolfin.FunctionSpace(mesh, T_e)
        Vbar = dolfin.FunctionSpace(mesh, Vbare) if self.periodic is None \
            else dolfin.FunctionSpace(mesh, Vbare,
                                      constrained_domain=self.periodic)
        self.Vbar = Vbar
        FuncSpace_adv = {'FuncSpace_local': self.V,
                         'FuncSpace_lambda': self.langrange_space,
                         'FuncSpace_bar': self.Vbar}
        self.forms_pde_map = \
            geopart.composition.incompressible.FormsPDEMapExtended(
                mesh, FuncSpace_adv)

        # This is messy
        self.T0_a = dolfin.Function(self.V)
        self.T0_abar = dolfin.Function(self.Vbar)
        self.dTh0 = dolfin.Function(self.V)
        self.dTh00 = dolfin.Function(self.V)

        self.prop_idxs = \
            numpy.array([self.property_idx, self.property_idx + 1],
                        dtype=numpy.uintp)

        self.solver = None

    def ufl_element(self, mesh: dolfin.Mesh):
        k = self.degree()
        Ve = ufl.FiniteElement("DG", mesh.ufl_cell(), k)
        # Vbare = ufl.FiniteElement("CG", mesh.ufl_cell(), k)["facet"]
        Vbare = ufl.FiniteElement("DGT", mesh.ufl_cell(), k)
        return Ve, Vbare

    def function_space(self):
        return self.V, self.Vbar

    def create_solution_variable(self, W: dolfin.FunctionSpace):
        return dolfin.Function(W[0]), dolfin.Function(W[1])

    @geopart.timings.apply_dolfin_timer
    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        mesh = W[0].mesh()
        Tstar, _ = Ustar

        Wh = self.function_space()
        funcspace_dict = {'FuncSpace_local': Wh[0],
                          'FuncSpace_bar': Wh[1]}

        alpha = dolfin.Constant(20.0 * self.degree() ** 2)
        theta_d = dolfin.Constant(1.0)
        T0_d, T0_dbar = Ustar
        forms_pde = FormsDiffusion(mesh, funcspace_dict).forms_theta(
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
        self.dTh0.vector()[:] = (T.vector() - Tstar.vector())/dt

        # Leopart indexes the first step as step = 1
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
    def project_advection(self, Ustar: dolfin.Function,
                          weak_bcs, model=None) -> None:
        Tstar, Tstarbar = Ustar
        if self.pde_projection is None:
            forms_pde_map = self.forms_pde_map

            psi_star = self.T0_a + self.dt_ufl * (
                    (1 - self.theta_L) * self.dTh00 + self.theta_L * self.dTh0)
            forms_pde = forms_pde_map.forms_theta_pseudo_compressible_linear(
                None, self.u_vec, self.dt_ufl, theta_map=self.theta,
                rho=model.rho,
                theta_L=self.theta_L, dpsi0=None, dpsi00=None,
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
        rho = model.rho

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
        F += rho * dTdt * s * dx

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
                   model: HeatModel, solve_as_linear_problem=False,
                   assemble_lhs=True) -> None:
        if solve_as_linear_problem:
            if not hasattr(self, "assembler_diffusion"):
                (Fr, J), strong_bcs = self.generate_diffusion_forms(
                    W, U, Ustar, weak_bcs, model)
                self.strong_bcs_diff = strong_bcs
                self.strong_bcs_hom = [dolfin.DirichletBC(bc)
                                       for bc in self.strong_bcs_diff]
                for bc in self.strong_bcs_hom:
                    bc.homogenize()

                self.assembler_diffusion = leopart.AssemblerStaticCondensation(
                    J[0][0], J[0][1],
                    J[1][0], J[1][1],
                    -Fr[0], -Fr[1]
                )

                solver = dolfin.PETScKrylovSolver()
                solver.set_options_prefix("hdg_heat_diff_")
                dolfin.PETScOptions.set("hdg_heat_diff_ksp_type", "preonly")
                dolfin.PETScOptions.set("hdg_heat_diff_pc_type", "lu")
                dolfin.PETScOptions.set(
                    "hdg_heat_diff_pc_factor_mat_solver_type", "mumps")
                solver.set_from_options()
                self.solver = solver

            U[0].vector().zero()
            U[1].vector().zero()
            for bc in self.strong_bcs_diff:
                # bc.apply(U[0].vector())
                # bc.apply(U[1].vector())
                if U[0].function_space().contains(
                        dolfin.FunctionSpace(bc.function_space())):
                    bc.apply(U[0].vector())
                if U[1].function_space().contains(
                        dolfin.FunctionSpace(bc.function_space())):
                    bc.apply(U[1].vector())

            dU = [None] * len(U)
            dU[0] = U[0].copy(deepcopy=True)
            dU[1] = U[1].copy(deepcopy=True)
            A, b = mats

            if assemble_lhs:
                self.assembler_diffusion.assemble_global(A, b)
                self.solver.set_operator(A)
                self.solver.set_reuse_preconditioner(False)
                for bc in self.strong_bcs_hom:
                    bc.apply(A, b)
            else:
                self.solver.set_reuse_preconditioner(True)
                self.assembler_diffusion.assemble_global_rhs(b)
                for bc in self.strong_bcs_hom:
                    bc.apply(b)

            dUh, dUhbar = dU
            self.solver.solve(dUhbar.vector(), b)
            self.assembler_diffusion.backsubstitute(
                dUhbar._cpp_object, dUh._cpp_object)

            U[0].vector().axpy(1.0, dUh.vector())
            U[1].vector().axpy(1.0, dUhbar.vector())
        else:
            if self.solver is None:
                (Fr, J), strong_bcs = self.generate_diffusion_forms(
                    W, U, Ustar, weak_bcs, model)
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


class HeatLeastSquaresHDG2(HeatLeastSquaresHDG1):

    def degree(self):
        return 2


class HeatLeastSquaresNonlinearDiffusionHDG1(
        HeatPDEConstrainedNonlinearDiffusionHDG1):

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
                          bcs, model=None) -> None:
        Tstar, Tstarbar = Ustar
        self.l2p.project(Tstar.cpp_object())


class HeatLeastSquaresNonlinearDiffusionHDG2(
    HeatLeastSquaresNonlinearDiffusionHDG1):

    def degree(self):
        return 2


class HeatCG1(geopart.particleprojection.AbstractParticleProjectionScheme):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        solver = dolfin.PETScKrylovSolver()
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "lu")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        solver.set_from_options()

        self.solver = solver
        self.dt_ufl = dt_ufl
        self.u_vec = u_vec
        self.theta = dolfin.Constant(0.5)

        mesh = ptcls.mesh()
        self.V = dolfin.FunctionSpace(mesh, self.ufl_element(mesh))

        # TODO: this is messy
        self.T0_a = dolfin.Function(self.V)
        self.system_assembler = None

    def degree(self):
        return 1

    def ufl_element(self, mesh: dolfin.Mesh):
        return ufl.FiniteElement("CG", mesh.ufl_cell(), self.degree())

    def function_space(self):
        return self.V

    def create_solution_variable(self, W: dolfin.FunctionSpace) -> dolfin.Function:
        return dolfin.Function(W)

    def project_advection(self, Ustar: dolfin.Function,
                          bcs) -> None:
        Ustar.vector()[:] = self.T0_a.vector()

    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        dT = dolfin.TrialFunction(W)
        T0 = Ustar
        v = dolfin.TestFunction(W)
        kappa = model.kappa
        Q = model.Q

        dTdt = (dT - T0) / self.dt_ufl
        Tth = self.theta*dT + (1 - self.theta)*T0

        F = dTdt * v * dx + dot(self.u_vec, grad(Tth))*v*dx \
            + dot(kappa*grad(Tth), grad(v))*dx - Q * v * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= bc.get_function() * v * bc.get_boundary()

        a, L = ufl.lhs(F), ufl.rhs(F)
        return a, L

    def assemble(self, forms: typing.Sequence[ufl.Form],
                 mats: typing.Tuple[dolfin.PETScMatrix, dolfin.PETScVector],
                 bcs: typing.Sequence[dolfin_dg.operators.DGBC]):
        if self.system_assembler is None:
            a, L = forms
            V = L.arguments()[0].function_space()
            strong_bcs = [
                dolfin.DirichletBC(V, bc.get_function(),
                                   bc.get_boundary().subdomain_data(),
                                   bc.get_boundary().subdomain_id())
                for bc in bcs if isinstance(bc, dolfin_dg.DGDirichletBC)]
            self.system_assembler = dolfin.SystemAssembler(a, L, strong_bcs)
        A, b = mats
        self.system_assembler.assemble(A, b)

    def solve_diffusion(self, W: dolfin.FunctionSpace, T: dolfin.Function,
                   T0: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        # Initial Stokes solve
        a, L = self.generate_diffusion_forms(W, T, T0, weak_bcs, model)
        A, b = mats

        T.vector().zero()  # Homogenise the problem
        self.assemble((a, L), (A, b), weak_bcs)

        self.solver.set_operator(A)
        self.solver.solve(T.vector(), b)

    def get_heat(self, U: dolfin.Function):
        return U

    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        self.T0_a.vector()[:] = U.vector()


class HeatCG2(HeatCG1):

    def degree(self):
        return 2


class HeatCG3(HeatCG1):

    def degree(self):
        return 3


class CustomProblem(dolfin.NonlinearProblem):

    def __init__(self, F, J, bcs):
        self._F = F
        self._J = J
        self.bcs = bcs
        super().__init__()

    def F(self, b, x):
        dolfin.assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        dolfin.assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

class HeatNonlinearDiffusionCG1(HeatCG1):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        self.newton_solver = None
        super().__init__(ptcls, u_vec, dt_ufl, property_idx, zeta_val=zeta_val)

    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        T, T0 = U, Ustar
        v = dolfin.TestFunction(W)
        kappa = model.kappa
        Q = model.Q
        rho = model.rho

        dTdt = (T - T0) / self.dt_ufl
        Tth = self.theta*T + (1 - self.theta)*T0

        F = rho * dTdt * v * dx + rho * dot(self.u_vec, grad(Tth))*v*dx \
            + dot(kappa*grad(Tth), grad(v))*dx - Q * v * dx

        for bc in weak_bcs:
            if isinstance(bc, dolfin_dg.DGDirichletBC):
                pass
            if isinstance(bc, dolfin_dg.DGNeumannBC):
                F -= bc.get_function() * v * bc.get_boundary()

        J = dolfin.derivative(F, T)
        return F, J

    def solve_diffusion(self, W: dolfin.FunctionSpace, T: dolfin.Function,
                   T0: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        # Initial Stokes solve
        if self.newton_solver is None:
            F, J = self.generate_diffusion_forms(W, T, T0, weak_bcs, model)
            strong_bcs = [
                dolfin.DirichletBC(W, bc.get_function(),
                                   bc.get_boundary().subdomain_data(),
                                   bc.get_boundary().subdomain_id())
                for bc in weak_bcs if isinstance(bc, dolfin_dg.DGDirichletBC)]

            # dolfin.solve(F == 0, T, strong_bcs, J=J)
            self.problem = CustomProblem(F, J, strong_bcs)
            self.newton_solver = dolfin.NewtonSolver()

        self.newton_solver.solve(self.problem, T.vector())

    def get_heat(self, U: dolfin.Function):
        return U

    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        self.T0_a.vector()[:] = U.vector()


class HeatNonlinearDiffusionCG2(HeatNonlinearDiffusionCG1):

    def degree(self):
        return 2


class HeatDG1(geopart.particleprojection.AbstractParticleProjectionScheme):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        solver = dolfin.PETScKrylovSolver()
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "lu")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        solver.set_from_options()

        self.solver = solver
        self.dt_ufl = dt_ufl
        self.u_vec = u_vec
        self.theta = dolfin.Constant(0.5)

        mesh = ptcls.mesh()
        self.V = dolfin.FunctionSpace(mesh, self.ufl_element(mesh))

        # TODO: this is messy
        self.T0_a = dolfin.Function(self.V)
        self.system_assembler = None
        self.newton_solver = None

    def degree(self):
        return 1

    def ufl_element(self, mesh: dolfin.Mesh):
        return ufl.FiniteElement("DG", mesh.ufl_cell(), self.degree())

    def function_space(self):
        return self.V

    def create_solution_variable(self, W: dolfin.FunctionSpace) -> dolfin.Function:
        return dolfin.Function(W)

    def project_advection(self, Ustar: dolfin.Function,
                          bcs) -> None:
        Ustar.vector()[:] = self.T0_a.vector()

    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        mesh = W.mesh()
        s = dolfin.TestFunction(W)
        T, T0 = U, Ustar
        v = dolfin.TestFunction(W)
        # kappa = model.kappa
        # Q = model.Q
        # rho = model.rho
        # Tth = self.theta*T + (1-self.theta)*T0

        f = model.Q

        # -- Second order
        def F_v(u, grad_u):
            return model.kappa*grad_u

        eo = dolfin_dg.EllipticOperator(mesh, W, weak_bcs, F_v)
        F = eo.generate_fem_formulation(T, s)

        # -- First order
        b = self.u_vec
        def F_c(u):
            if hasattr(u, "side"):
                return b(u.side())*u
            return b * u

        def flux_evs(u, n):
            if hasattr(u, "side"):
                return dot(b(u.side()), n)
            return dot(b, n)
        flux_function = dolfin_dg.LocalLaxFriedrichs(flux_evs)
        ho = dolfin_dg.HyperbolicOperator(
            mesh, W, weak_bcs, F_c, flux_function)
        F += ho.generate_fem_formulation(T, s)

        # -- Source
        F -= f * s * dx

        # # -- theta scheme
        # F = ufl.replace(F, {T: Tth})

        dTdt = (T - T0) / self.dt_ufl
        F += dTdt * s * dx

        J = dolfin.derivative(F, T)
        return F, J

    def solve_diffusion(self, W: dolfin.FunctionSpace, T: dolfin.Function,
                   T0: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        # Initial Stokes solve
        if self.newton_solver is None:
            F, J = self.generate_diffusion_forms(W, T, T0, weak_bcs, model)
            strong_bcs = []

            # dolfin.solve(F == 0, T, strong_bcs, J=J)
            self.problem = CustomProblem(F, J, strong_bcs)
            self.newton_solver = dolfin.NewtonSolver()

        self.newton_solver.solve(self.problem, T.vector())

    def get_heat(self, U: dolfin.Function):
        return U

    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        self.T0_a.vector()[:] = U.vector()



class HeatSplitDG1(geopart.particleprojection.AbstractParticleProjectionScheme):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        solver = dolfin.PETScKrylovSolver()
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "lu")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        solver.set_from_options()

        self.solver = solver
        self.dt_ufl = dt_ufl
        self.u_vec = u_vec
        self.theta = dolfin.Constant(0.5)

        mesh = ptcls.mesh()
        self.V = dolfin.FunctionSpace(mesh, self.ufl_element(mesh))

        # TODO: this is messy
        self.T0_a = dolfin.Function(self.V)
        self.system_assembler = None
        self.newton_solver = None
        self.newton_solver_a = None

    def degree(self):
        return 1

    def ufl_element(self, mesh: dolfin.Mesh):
        return ufl.FiniteElement("DG", mesh.ufl_cell(), self.degree())

    def function_space(self):
        return self.V

    def create_solution_variable(self, W: dolfin.FunctionSpace) -> dolfin.Function:
        return dolfin.Function(W)

    def generate_advection_forms(self, Ustar, weak_bcs):
        mesh = Ustar.function_space().mesh()
        W = Ustar.function_space()
        s = dolfin.TestFunction(W)
        Tstar = Ustar

        # -- First order
        b = self.u_vec
        def F_c(u):
            if hasattr(u, "side"):
                return b(u.side())*u
            return b * u

        def flux_evs(u, n):
            if hasattr(u, "side"):
                return dot(b(u.side()), n)
            return dot(b, n)
        flux_function = dolfin_dg.LocalLaxFriedrichs(flux_evs)
        ho = dolfin_dg.HyperbolicOperator(
            mesh, W, weak_bcs, F_c, flux_function)
        F = ho.generate_fem_formulation(Tstar, s)

        dTdt = (Tstar - self.T0_a) / self.dt_ufl
        F += dTdt * s * dx

        J = dolfin.derivative(F, Tstar)

        return F, J

    def project_advection(self, Ustar: dolfin.Function,
                          weak_bcs) -> None:
        if self.newton_solver_a is None:
            F, J = self.generate_advection_forms(Ustar, weak_bcs)

            self.problem_a = CustomProblem(F, J, [])
            self.newton_solver_a = dolfin.NewtonSolver()
        self.newton_solver_a.solve(self.problem_a, Ustar.vector())


    def generate_diffusion_forms(self, W: dolfin.FunctionSpace, U: dolfin.Function,
                                 Ustar: dolfin.Function,
                                 weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                                 model: HeatModel):
        mesh = W.mesh()
        s = dolfin.TestFunction(W)
        T, T0 = U, Ustar
        v = dolfin.TestFunction(W)
        kappa = model.kappa
        Q = model.Q
        rho = model.rho
        Tth = self.theta*T + (1-self.theta)*T0

        f = model.Q

        # -- Second order
        def F_v(u, grad_u):
            return model.kappa*grad_u

        eo = dolfin_dg.EllipticOperator(mesh, W, weak_bcs, F_v)
        F = eo.generate_fem_formulation(T, s)

        # -- Source
        F -= f * s * dx

        # # -- theta scheme
        # F = ufl.replace(F, {T: Tth})

        dTdt = (T - T0) / self.dt_ufl
        F += dTdt * s * dx

        J = dolfin.derivative(F, T)
        return F, J

    def solve_diffusion(self, W: dolfin.FunctionSpace, T: dolfin.Function,
                   T0: dolfin.Function,
                   mats: typing.Sequence[typing.Union[
                       dolfin.Vector, dolfin.Matrix]],
                   weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
                   model: HeatModel) -> None:
        # Initial Stokes solve
        if self.newton_solver is None:
            F, J = self.generate_diffusion_forms(W, T, T0, weak_bcs, model)
            strong_bcs = []

            self.problem = CustomProblem(F, J, strong_bcs)
            self.newton_solver = dolfin.NewtonSolver()

        self.newton_solver.solve(self.problem, T.vector())

    def get_heat(self, U: dolfin.Function):
        return U

    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int, dt: float) -> None:
        self.T0_a.vector()[:] = U.vector()



class HeatPDEConstrainedThetaSchemeHDG1(HeatPDEConstrainedHDG1):

    def __init__(self, ptcls: leopart.particles,
                 u_vec: dolfin.Function,
                 dt_ufl: dolfin.Constant,
                 property_idx: int,
                 zeta_val: float = 0.0):
        super().__init__(ptcls, u_vec, dt_ufl, property_idx, zeta_val)
        self.theta.assign(0.5)

    @geopart.timings.apply_dolfin_timer
    def project_advection(self, Ustar: dolfin.Function,
                          weak_bcs, model=None) -> None:
        Tstar, Tstarbar = Ustar
        if self.pde_projection is None:
            forms_pde_map = self.forms_pde_map

            psi_star = self.T0_a
            forms_pde = forms_pde_map.forms_theta_pseudo_compressible_linear(
                None, self.u_vec, self.dt_ufl, theta_map=self.theta,
                rho=model.rho,
                theta_L=self.theta_L, dpsi0=None, dpsi00=None,
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
        rho = model.rho

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
        Tth = self.theta*T + (1-self.theta)*Tstar
        F = ufl.replace(F, {T: Tth})
        F += rho * dTdt * s * dx

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
                   model: HeatModel, solve_as_linear_problem=False,
                   assemble_lhs=True) -> None:
        if solve_as_linear_problem:
            if not hasattr(self, "assembler_diffusion"):
                (Fr, J), strong_bcs = self.generate_diffusion_forms(
                    W, U, Ustar, weak_bcs, model)
                self.strong_bcs_diff = strong_bcs
                self.strong_bcs_hom = [dolfin.DirichletBC(bc)
                                       for bc in self.strong_bcs_diff]
                for bc in self.strong_bcs_hom:
                    bc.homogenize()

                self.assembler_diffusion = leopart.AssemblerStaticCondensation(
                    J[0][0], J[0][1],
                    J[1][0], J[1][1],
                    -Fr[0], -Fr[1]
                )

                solver = dolfin.PETScKrylovSolver()
                solver.set_options_prefix("hdg_heat_diff_")
                dolfin.PETScOptions.set("hdg_heat_diff_ksp_type", "preonly")
                dolfin.PETScOptions.set("hdg_heat_diff_pc_type", "lu")
                dolfin.PETScOptions.set(
                    "hdg_heat_diff_pc_factor_mat_solver_type", "mumps")
                solver.set_from_options()
                self.solver = solver

            U[0].vector().zero()
            U[1].vector().zero()
            for bc in self.strong_bcs_diff:
                # bc.apply(U[0].vector())
                # bc.apply(U[1].vector())
                if U[0].function_space().contains(
                        dolfin.FunctionSpace(bc.function_space())):
                    bc.apply(U[0].vector())
                if U[1].function_space().contains(
                        dolfin.FunctionSpace(bc.function_space())):
                    bc.apply(U[1].vector())

            dU = [None] * len(U)
            dU[0] = U[0].copy(deepcopy=True)
            dU[1] = U[1].copy(deepcopy=True)
            A, b = mats

            if assemble_lhs:
                self.assembler_diffusion.assemble_global(A, b)
                self.solver.set_operator(A)
                self.solver.set_reuse_preconditioner(False)
                for bc in self.strong_bcs_hom:
                    bc.apply(A, b)
            else:
                self.solver.set_reuse_preconditioner(True)
                self.assembler_diffusion.assemble_global_rhs(b)
                for bc in self.strong_bcs_hom:
                    bc.apply(b)

            dUh, dUhbar = dU
            self.solver.solve(dUhbar.vector(), b)
            self.assembler_diffusion.backsubstitute(
                dUhbar._cpp_object, dUh._cpp_object)

            U[0].vector().axpy(1.0, dUh.vector())
            U[1].vector().axpy(1.0, dUhbar.vector())
        else:
            if self.solver is None:
                (Fr, J), strong_bcs = self.generate_diffusion_forms(
                    W, U, Ustar, weak_bcs, model)
                solver = \
                    dolfin_dg.dolfin.hdg_newton.StaticCondensationNewtonSolver(
                        Fr, J, strong_bcs)
                self.solver = solver
            T, Tbar = U
            self.solver.solve(T, Tbar)

    @geopart.timings.apply_dolfin_timer
    def update_field_and_increment_particles(
            self, U, Ustar, theta_p: float, step: int,
            dt: float) -> None:
        T = U[0]
        Tstar = Ustar[0]

        self.T0_a.vector()[:] = T.vector()

        # Leopart indexes the first step as step = 1
        self.ptcls.interpolate(T, self.prop_idxs[0])


class HeatPDEConstrainedThetaSchemeHDG2(HeatPDEConstrainedThetaSchemeHDG1):

    def degree(self):
        return 2