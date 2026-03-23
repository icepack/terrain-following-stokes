from numpy import pi as π
import firedrake
from firedrake import Constant


def linear(ξ):
    b_0 = Constant(-1.0)
    b_1 = Constant(-0.5)
    s_0 = Constant(1.0)
    s_1 = Constant(0.5)

    bed = (1 - ξ) * b_0 + ξ * b_1
    surface = (1 - ξ) * s_0 + ξ * s_1
    thickness = surface - bed
    return bed, thickness


def wavy(ξ, wavenumber=1.0, phase=0.0):
    s_0 = Constant(1.0)
    δs = Constant(0.25)
    k = Constant(wavenumber)
    φ = Constant(phase)
    surface = s_0 + δs * firedrake.cos(2 * π * (k * ξ + φ))
    bed = Constant(0.0) * ξ
    thickness = surface - bed
    return bed, thickness

