import firedrake
from firedrake import (
    Constant, inner, outer, dot, sym, grad, div, dx, avg, jump, dS_h, dS_v,
    ds_b, ds_t, ds_v,
)
import ufl



def coordinate_transformation_derivatives(b, h):
    mesh = ufl.domain.extract_unique_domain(b)
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

    μ = Constant(kwargs["viscosity"])
    τ = 2 * μ * ε

    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    return u, p, ε, τ, b, h, J, J_inv, n, mesh


def cell_free_energy_rate(**kwargs):
    u, p, ε, τ, _, h, J, _, _, mesh = _get_fields(**kwargs)

    g = kwargs["gravity"]
    f = Constant((0,) * (mesh.geometric_dimension - 1) + (-g,))

    return (0.5 * h * inner(τ, ε) - p * div(h * u) - h * inner(f, dot(J, u))) * dx


def boundary_measure(ids):
    numeric_ids = tuple(set(ids) - {"bottom", "top"})
    ds = ds_v(numeric_ids)
    if "bottom" in ids:
        ds += ds_b
    if "top" in ids:
        ds += ds_t
    return ds


def facet_free_energy_rate(**kwargs):
    u, p, ε, τ, _, h, J, J_inv, n, mesh = _get_fields(**kwargs)

    α = Constant(kwargs.get("penalty", 100.0))
    μ = Constant(kwargs["viscosity"])
    γ = firedrake.CellSize(mesh)

    u_n = outer(dot(J, u), dot(n, J_inv))
    δu_n = u_n("+") + u_n("-")
    δu = jump(dot(J, u))

    dS = dS_h + dS_v
    fpower = (-inner(avg(h * τ), δu_n) + avg(p) * jump(h * u, n)) * dS
    fpenalty = α * μ / (2 * avg(γ)) * inner(δu, δu) * avg(h) * dS

    ds = boundary_measure(kwargs["dirichlet_ids"])
    bpower = (-inner(τ, u_n) + p * inner(u, n)) * h * ds
    bpenalty = α * μ / (2 * γ) * inner(dot(J, u), dot(J, u)) * h * ds

    return fpower + fpenalty + bpower + bpenalty


def free_energy_rate(**kwargs):
    return cell_free_energy_rate(**kwargs) + facet_free_energy_rate(**kwargs)
