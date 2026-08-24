import firedrake
from firedrake import (
    Constant, inner, outer, sym, grad, div, dx, avg, jump, dS_h, dS_v,
)
import ufl
from .common import boundary_measure


def _get_fields(**kwargs):
    u = kwargs["velocity"]
    p = kwargs["pressure"]

    ε = sym(grad(u))
    if kwargs.get("form", "primal") == "primal":
        μ = Constant(kwargs["viscosity"])
        τ = 2 * μ * ε
    else:
        τ = kwargs["stress"]

    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    return u, p, ε, τ, n, mesh


def cell_free_energy_rate(**kwargs):
    u, p, ε, τ, _, mesh = _get_fields(**kwargs)

    g = kwargs["gravity"]
    f = Constant((0,) * (mesh.geometric_dimension - 1) + (-g,))

    if kwargs.get("form", "primal") == "primal":
        return (0.5 * inner(τ, ε) - p * div(u) - inner(f, u)) * dx

    μ = kwargs["viscosity"]
    return (inner(τ, τ) / (4 * μ) - inner(τ, ε) + p * div(u) + inner(f, u)) * dx


def facet_free_energy_rate(**kwargs):
    u, p, _, τ, n, mesh = _get_fields(**kwargs)

    α = Constant(kwargs.get("penalty", 100.0))
    μ = Constant(kwargs["viscosity"])
    γ = firedrake.CellSize(mesh)

    u_n = sym(outer(u, n))

    dS = dS_h + dS_v
    # TODO: Try alternative forms for this
    power = (-inner(avg(τ), u_n("+") + u_n("-")) + avg(p) * jump(u, n)) * dS
    penalty = α * μ / (2 * avg(γ)) * inner(jump(u), jump(u)) * dS

    if "dirichlet_ids" in kwargs:
        ds_dirichlet = boundary_measure(kwargs["dirichlet_ids"])
        power += (-inner(τ, u_n) + p * inner(u, n)) * ds_dirichlet
        penalty += α * μ / (2 * γ) * inner(u, u) * ds_dirichlet

    if "robin_ids" in kwargs:
        ds_robin = boundary_measure(kwargs["robin_ids"])
        power += (-inner(τ, outer(n, n)) + p) * inner(u, n) * ds_robin
        penalty += α * μ / (2 * γ) * inner(u, n)**2 * ds_robin

    return power + penalty


def free_energy_rate(**kwargs):
    G_cells = cell_free_energy_rate(**kwargs)
    G_facets = facet_free_energy_rate(**kwargs)
    if kwargs.get("form", "primal") == "primal":
        return G_cells + G_facets
    return G_cells - G_facets
