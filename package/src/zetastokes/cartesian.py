import firedrake
from firedrake import (
    Constant, inner, outer, sym, grad, div, dx, avg, jump, dS_h, dS_v,
    ds_b, ds_t, ds_v
)
import ufl


def _get_fields(**kwargs):
    u = kwargs["velocity"]
    p = kwargs["pressure"]

    μ = Constant(kwargs["viscosity"])
    ε = sym(grad(u))
    τ = 2 * μ * ε

    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    return u, p, ε, τ, n, mesh


def cell_free_energy_rate(**kwargs):
    u, p, ε, τ, _, mesh = _get_fields(**kwargs)

    g = kwargs["gravity"]
    f = Constant((0,) * (mesh.geometric_dimension - 1) + (-g,))

    return (0.5 * inner(τ, ε) - p * div(u) - inner(f, u)) * dx


def boundary_measure(ids):
    numeric_ids = tuple(set(ids) - {"bottom", "top"})
    ds = ds_v(numeric_ids)
    if "bottom" in ids:
        ds += ds_b
    if "top" in ids:
        ds += ds_t
    return ds


def facet_free_energy_rate(**kwargs):
    u, p, _, τ, n, mesh = _get_fields(**kwargs)

    α = Constant(kwargs.get("penalty", 100.0))
    μ = Constant(kwargs["viscosity"])
    γ = firedrake.CellSize(mesh)

    u_n = sym(outer(u, n))

    dS = dS_h + dS_v
    fpower = (-inner(avg(τ), u_n("+") + u_n("-")) + avg(p) * jump(u, n)) * dS
    fpenalty = α * μ / (2 * avg(γ)) * inner(jump(u), jump(u)) * dS

    ds = boundary_measure(kwargs["dirichlet_ids"])
    bpower = (-inner(τ, u_n) + p * inner(u, n)) * ds
    bpenalty = α * μ / (2 * γ) * inner(u, u) * ds
    return fpower + fpenalty + bpower + bpenalty


def free_energy_rate(**kwargs):
    return cell_free_energy_rate(**kwargs) + facet_free_energy_rate(**kwargs)
