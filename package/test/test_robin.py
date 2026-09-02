import numpy as np
import firedrake
from firedrake import assemble, inner, Constant, ds_b, ds_t, ds_v, dx
import ufl
from zetastokes.terrain_following import free_energy_rate


lx = 5.0
constants = {
    "viscosity": 1.0, "density": 1.0, "gravity": 9.81, "skin_depth": 0.125
}
pparams = {"form_compiler_parameters": {"quadrature_degree": 8}}
sparams = {"pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def bed(x):
    b_0 = Constant(0.0)
    b_1 = Constant(0.25)
    return (1 - x) * b_0 + x * b_1


def thickness(x):
    h_0 = Constant(1.0)
    h_1 = Constant(0.5)
    return (1 - x) * h_0 + x * h_1


def setup(mesh, degree):
    u_elt = firedrake.FiniteElement("DQ", "quadrilateral", degree + 1)
    p_elt = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    V = firedrake.VectorFunctionSpace(mesh, u_elt)
    Q = firedrake.FunctionSpace(mesh, p_elt)
    Z = V * Q
    return firedrake.Function(Z)


def solve_terrain_following(mesh, degree):
    z = setup(mesh, degree)
    u, p = firedrake.split(z)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    b, h = bed(x), thickness(x)

    fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h}
    boundary_data = {"robin_ids": [1, 2, "bottom", "top"]}
    G_stokes = free_energy_rate(**fields, **constants, **boundary_data)

    u_t = Constant((1.0, 0.0))
    u_b = Constant((-1.0, 0.0))
    μ = Constant(constants["viscosity"])
    λ = Constant(constants["skin_depth"])
    G_forcing = (
        0.5 * μ / λ * inner(u - u_t, u - u_t) * h * ds_t +
        0.5 * μ / λ * inner(u - u_b, u - u_b) * h * ds_b
    )
    G = G_stokes + G_forcing

    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, **pparams)
    return z


def test_robin_bcs():
    nxs = np.array([8, 16, 32, 64, 128])
    errors = np.zeros_like(nxs, dtype=float)
    for index, nx in enumerate(nxs):
        interval = firedrake.UnitIntervalMesh(nx)
        mesh = firedrake.ExtrudedMesh(interval, nx)
        z = solve_terrain_following(mesh, 1)
        u, p = z.subfunctions

        n = firedrake.FacetNormal(mesh)
        length = assemble(Constant(1) * (ds_b(mesh) + ds_t(mesh) + ds_v(mesh)))
        flux_avg = assemble(abs(inner(u, n)) * (ds_b + ds_t + ds_v)) / length
        area = assemble(Constant(1) * dx(mesh))
        speed_avg = np.sqrt(assemble(inner(u, u) * dx) / area)
        errors[index] = flux_avg / speed_avg

    slope, intercept = np.polyfit(np.log(1 / nxs), np.log(errors), 1)
    print(f"log ∫|u · n| ds ~= {intercept:0.2f} + {slope:0.2f} * log(δx)")
    assert slope > 0.9
