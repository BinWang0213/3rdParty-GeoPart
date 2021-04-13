import dolfin
import leopart

import geopart.composition.incompressible


class LeastSquaresDG0(geopart.composition.incompressible.LeastSquaresDG0):

    def project(self, phi: dolfin.Function, rho: dolfin.Expression) -> None:
        super().project(phi)


class LeastSquaresDG1(LeastSquaresDG0):

    def poly_order(self) -> int:
        return 1


class LeastSquaresDG2(LeastSquaresDG0):

    def poly_order(self) -> int:
        return 2


class PDEConstrainedDG0(geopart.composition.incompressible.PDEConstrainedDG0):

    def project(self, phi: dolfin.Function, rho: dolfin.Expression) -> None:
        if self.pde_projection is None:
            theta = dolfin.Constant(0.5)
            forms_pde_map = self.forms_pde_map
            forms_pde = forms_pde_map.forms_theta_linear(
                phi, self.u, self.dt_ufl, theta,
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
        # self.pde_projection.solve_problem(self.phibar_h.cpp_object(),
        #                                   phi.cpp_object(),
        #                                   self.lambda_h.cpp_object(),
        #                                   "mumps", "default")
        self.pde_projection.solve_problem(self.phibar_h.cpp_object(),
                                          phi.cpp_object(),
                                          "superlu_dist", "default")


class PDEConstrainedDG1(PDEConstrainedDG0):
    def degree(self) -> int:
        return 1


class PDEConstrainedDG2(PDEConstrainedDG0):
    def degree(self) -> int:
        return 2


class PDEConstrainedDRhoPhiDG0(
        geopart.composition.incompressible.PDEConstrainedDG0):

    def project(self, phi: dolfin.Function, rho: dolfin.Expression) -> None:
        if self.pde_projection is None:
            theta = dolfin.Constant(0.5)
            forms_pde_map = self.forms_pde_map
            forms_pde = forms_pde_map.forms_theta_pseudo_compressible_linear(
                phi, self.u, self.dt_ufl, theta, rho,
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
        # self.pde_projection.solve_problem(self.phibar_h.cpp_object(),
        #                                   phi.cpp_object(),
        #                                   self.lambda_h.cpp_object(),
        #                                   "mumps", "default")
        self.pde_projection.solve_problem(self.phibar_h.cpp_object(),
                                          phi.cpp_object(),
                                          "superlu_dist", "default")


class PDEConstrainedDRhoPhiDG1(PDEConstrainedDRhoPhiDG0):

    def degree(self) -> int:
        return 1


class PDEConstrainedDRhoPhiDG2(PDEConstrainedDRhoPhiDG0):

    def degree(self) -> int:
        return 2


class PDEConstrainedDRhoPhiDG3(PDEConstrainedDRhoPhiDG0):

    def degree(self) -> int:
        return 3
