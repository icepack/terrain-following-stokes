import firedrake
from firedrake import (
    Constant, inner, outer, sym, grad, div, dx, avg, jump, dS_h, dS_v,
)
from irksome import Dt
from .common import boundary_measure, get_test_function
from ufl.domain import extract_unique_domain


def _get_fields(**kwargs):
    u = kwargs["velocity"]
    p = kwargs["pressure"]

    ε = sym(grad(u))
    if kwargs.get("form", "primal") == "primal":
        μ = Constant(kwargs["viscosity"])
        τ = 2 * μ * ε
    else:
        τ = kwargs["stress"]

    mesh = extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    return u, p, ε, τ, n, mesh


def cell_free_energy_rate(**kwargs):
    u, p, ε, τ, _, mesh = _get_fields(**kwargs)

    ρ, g = map(kwargs.get, ["density", "gravity"])
    f = firedrake.as_vector([0] * (mesh.geometric_dimension - 1) + [-ρ * g])

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


def density_equation(**kwargs):
    field_names = ["density", "velocity"]
    ρ, u = map(kwargs.get, field_names)

    mesh = extract_unique_domain(ρ)
    dim = mesh.geometric_dimension
    n = firedrake.FacetNormal(mesh)

    v = Constant([0] * dim)
    if kwargs.get("frame", True):
        raise NotImplementedError("IOU 1 PDE")

    φ = get_test_function(ρ)
    F_cells = (Dt(ρ) * φ - ρ * inner(u, grad(φ))) * dx

    dS = dS_h + dS_v
    f = ρ * firedrake.max_value(0, inner(u - v, n))
    F_facets = jump(f) * jump(φ) * dS

    return F_cells + F_facets
