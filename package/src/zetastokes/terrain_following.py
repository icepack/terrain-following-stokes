import firedrake
from firedrake import (
    Constant, inner, outer, dot, sym, grad, div, dx, avg, jump, dS_h, dS_v
)
from irksome import Dt
from .common import boundary_measure, get_test_function
from ufl.domain import extract_unique_domain



def coordinate_transformation_derivatives(b, h):
    mesh = extract_unique_domain(b) or extract_unique_domain(h)
    d = mesh.geometric_dimension
    ζ = firedrake.SpatialCoordinate(mesh)[d - 1]
    σ = grad(b) + ζ * grad(h)
    if d == 2:
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
    u, p, ε, τ, _, h, J, J_inv, n, mesh = _get_fields(**kwargs)

    α = Constant(kwargs.get("penalty", 100.0))
    μ = Constant(kwargs["viscosity"])
    γ = firedrake.CellSize(mesh)

    ν = dot(n, J_inv)
    u_n = outer(dot(J, u), ν)
    δu_n = u_n("+") + u_n("-")
    δu = jump(dot(J, u))

    # TODO: Try alternative forms for this
    dS = dS_h + dS_v
    power = (-inner(avg(h * τ), δu_n) + avg(p) * jump(h * u, n)) * dS
    penalty = α * μ / (2 * avg(γ)) * inner(δu, δu) * avg(h) * dS

    if "dirichlet_ids" in kwargs:
        ds = boundary_measure(kwargs["dirichlet_ids"])
        power += (-inner(τ, u_n) + p * inner(u, n)) * h * ds
        penalty += α * μ / (2 * γ) * inner(dot(J, u), dot(J, u)) * h * ds

    if "robin_ids" in kwargs:
        ds = boundary_measure(kwargs["robin_ids"])
        power += (-inner(τ, outer(n, n)) + p) * inner(u, n) * h * ds
        penalty += α * μ / (2 * γ) * inner(u, n)**2 * h * ds

    return power + penalty


def free_energy_rate(**kwargs):
    G_cells = cell_free_energy_rate(**kwargs)
    G_facets = facet_free_energy_rate(**kwargs)
    if kwargs.get("form", "primal") == "primal":
        return G_cells + G_facets
    return G_cells - G_facets


def mass_balance(**kwargs):
    h = kwargs["thickness"]
    u = kwargs["velocity"]
    mesh = extract_unique_domain(h)
    n = firedrake.FacetNormal(mesh)
    φ = get_test_function(h)
    F_cells = (Dt(h) * φ - inner(h * u, grad(φ))) * dx
    u_n = firedrake.max_value(0, inner(u, n))
    ds = boundary_measure([1, 2])
    F_outflow = h * u_n * φ * ds
    return F_cells + F_outflow
