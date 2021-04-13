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
import geopart.energy.heat
import geopart.timings


class HeatCompressibleCG1(geopart.energy.heat.HeatNonlinearDiffusionCG1):

    def generate_diffusion_forms(
            self, W: dolfin.FunctionSpace, U: dolfin.Function,
            Ustar: dolfin.Function,
            weak_bcs: typing.Sequence[dolfin_dg.operators.DGBC],
            model: geopart.energy.heat.HeatModel):
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
