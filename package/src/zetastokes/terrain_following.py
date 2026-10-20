import firedrake
from firedrake import (
    Constant, inner, outer, dot, sym, grad, div, dx, dS_h, dS_v, avg, jump, sqrt
)
from irksome import Dt
from .common import horizontal_ds as ds_h, vertical_ds as ds_v, get_test_function
from ufl.domain import extract_unique_domain


def coordinate_surface_slope(b, h):
    mesh = extract_unique_domain(b) or extract_unique_domain(h)
    d = mesh.geometric_dimension
    ζ = firedrake.SpatialCoordinate(mesh)[d - 1]
    return grad(b) + ζ * grad(h)


def coordinate_transformation_derivatives(b, h):
    σ = coordinate_surface_slope(b, h)
    if σ.ufl_shape == (2,):
        J = [[1, 0], [σ[0], h]]
        J_inv = [[1, 0], [-σ[0] / h, 1 / h]]
    else:
        J = [[1, 0, 0], [0, 1, 0], [σ[0], σ[1], h]]
        J_inv = [[1, 0, 0], [0, 1, 0], [-σ[0] / h, -σ[1] / h, 1 / h]]

    return firedrake.as_tensor(J), firedrake.as_tensor(J_inv)


def _get_fields(**kwargs):
    u = kwargs["velocity"]
    p = kwargs["pressure"]
    b = kwargs["bed"]
    h = kwargs["thickness"]

    J, J_inv = coordinate_transformation_derivatives(b, h)
    grad_u = dot(grad(dot(J, u)), J_inv)
    ε = sym(grad_u)

    if kwargs.get("form", "primal") == "primal":
        μ = Constant(kwargs["viscosity"])
        τ = 2 * μ * ε
    else:
        τ = kwargs["stress"]

    mesh = extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    return u, p, ε, τ, b, h, J, J_inv, n, mesh


def cell_free_energy_rate(**kwargs):
    u, p, ε, τ, _, h, J, _, _, mesh = _get_fields(**kwargs)

    ρ, g = map(kwargs.get, ["density", "gravity"])
    f = firedrake.as_vector([0] * (mesh.geometric_dimension - 1) + [-ρ * g])
    G = (p * div(h * u) + h * inner(f, dot(J, u))) * dx

    if kwargs.get("form", "primal") == "primal":
        return 0.5 * h * inner(τ, ε) * dx - G

    μ = kwargs["viscosity"]
    return (h * inner(τ, τ) / (4 * μ) - h * inner(τ, ε)) * dx + G


def facet_free_energy_rate(**kwargs):
    u, p, ε, τ, b, h, J, J_inv, n, mesh = _get_fields(**kwargs)

    d = mesh.geometric_dimension
    α = Constant(kwargs.get("penalty", 100.0))
    μ = Constant(kwargs["viscosity"])

    λ = firedrake.CellVolume(mesh) / firedrake.FacetArea(mesh)
    Ju = dot(J, u)
    u_n = outer(Ju, dot(n, J_inv))
    δu = u_n("+") + u_n("-")

    traction = -inner(avg(τ), δu) * avg(h) * (dS_h + dS_v)
    pressure = avg(p) * jump(u, n) * avg(h) * (dS_h + dS_v)
    penalty = α * μ / (2 * avg(λ)) * inner(δu, δu) * avg(h) * (dS_h + dS_v)
    facet_form = traction + pressure + penalty

    dirichlet_ids = kwargs.get("dirichlet_ids", ())
    ds_d = ds_h(dirichlet_ids) + ds_v(dirichlet_ids)
    dtraction = -inner(τ, u_n) * h * ds_d
    dpressure = p * inner(u, n) * h * ds_d
    dpenalty = α * μ / (2 * λ) * inner(u_n, u_n) * h * ds_d
    dirichlet_form = dtraction + dpressure + dpenalty

    robin_ids = kwargs.get("robin_ids", ())
    ds_r = ds_h(robin_ids) + ds_v(robin_ids)
    ν = dot(n, J_inv)
    τ_nn = inner(ν, dot(τ, ν)) / inner(ν, ν)
    rtraction = -τ_nn * inner(u, n) * h * ds_r
    rpressure = p * inner(u, n) * h * ds_r
    rpenalty = α * μ / (2 * λ) * inner(u, n)**2 * h * ds_r
    robin_form = rtraction + rpressure + rpenalty

    return facet_form + dirichlet_form + robin_form


def free_energy_rate(**kwargs):
    G_cells = cell_free_energy_rate(**kwargs)
    G_facets = facet_free_energy_rate(**kwargs)
    if kwargs.get("form", "primal") == "primal":
        return G_cells + G_facets
    return G_cells - G_facets


def thickness_equation(**kwargs):
    h = kwargs["thickness"]
    u = kwargs["velocity"]
    mesh = extract_unique_domain(h)
    n = firedrake.FacetNormal(mesh)
    φ = get_test_function(h)
    F = (Dt(h) * φ - inner(h * u, grad(φ))) * dx
    u_n = firedrake.max_value(0, inner(u, n))

    if (outflow_ids := kwargs.get("outflow_ids")):
        F += h * u_n * φ * ds_v(outflow_ids)

    return F


def density_equation(**kwargs):
    field_names = ["density", "thickness", "velocity"]
    ρ, h, u = map(kwargs.get, field_names)

    mesh = extract_unique_domain(ρ)
    dim = mesh.geometric_dimension
    n = firedrake.FacetNormal(mesh)

    v = Constant([0] * dim)
    if kwargs.get("frame", True):
        ζ = firedrake.SpatialCoordinate(mesh)[dim - 1]
        v = firedrake.as_vector([0] * (dim - 1) + [ζ * Dt(h)])

    φ = get_test_function(ρ)
    F_cells = (Dt(h * ρ) * φ - h * ρ * inner(u, grad(φ))) * dx
    F_frame = ρ * inner(v, grad(φ)) * dx

    dS = dS_h + dS_v
    f = ρ * firedrake.max_value(0, inner(h * u - v, n))
    F_facets = jump(f) * jump(φ) * dS

    return F_cells + F_frame + F_facets
