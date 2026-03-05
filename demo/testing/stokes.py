import firedrake
from firedrake import (
    Constant, as_tensor, inner, outer, dot, sym, grad, div, dx, avg, jump
)
import ufl


def free_energy_rate_cartesian(**kwargs) -> firedrake.Form:
    u = kwargs["velocity"]
    p = kwargs["pressure"]

    μ = Constant(kwargs["viscosity"])
    ε = sym(grad(u))
    τ = 2 * μ * ε

    g = kwargs["gravity"]
    mesh = ufl.domain.extract_unique_domain(u)
    if mesh.geometric_dimension == 2:
        f = Constant((0, -g))
    elif mesh.geometric_dimension == 3:
        f = Constant((0, 0, -g))

    return (0.5 * inner(τ, ε) - p * div(u) - inner(f, u)) * dx


def free_energy_rate_cartesian_dg(**kwargs) -> firedrake.Form:
    G_cells = free_energy_rate_cartesian(**kwargs)

    u = kwargs["velocity"]
    p = kwargs["pressure"]

    μ = Constant(kwargs["viscosity"])
    ε = sym(grad(u))
    τ = 2 * μ * ε

    mesh = ufl.domain.extract_unique_domain(u)
    ids = kwargs["dirichlet_ids"]
    if mesh.extruded:
        ids.remove("bottom")
        dS = firedrake.dS_h + firedrake.dS_v
        ds = firedrake.ds_v(tuple(ids)) + firedrake.ds_b
    else:
        dS = firedrake.dS
        ds = firedrake.ds(tuple(ids))

    n = firedrake.FacetNormal(mesh)
    I = firedrake.Identity(mesh.geometric_dimension)

    u_n = sym(outer(u, n))
    G_power = -inner(avg(τ - p * I), u_n("+") + u_n("-")) * dS

    α = Constant(kwargs["penalty"])
    γ = avg(firedrake.CellSize(mesh))
    G_penalty = α * μ / (2 * γ) * inner(jump(u), jump(u)) * dS

    # TODO: add non-zero boundary velocity
    u_Γ = Constant((0,) * mesh.geometric_dimension)
    G_boundary_power = -inner(τ - p * I, sym(outer(u - u_Γ, n))) * ds
    G_boundary_penalty = α * μ / (2 * γ) * inner(u - u_Γ, u - u_Γ) * ds

    return G_cells + G_power + G_penalty + G_boundary_power + G_boundary_penalty


def coordinate_transformation_derivative(b, h):
    mesh = ufl.domain.extract_unique_domain(b)
    _, _, ζ = firedrake.SpatialCoordinate(mesh)
    σ = grad(b) + ζ * grad(h)
    if mesh.geometric_dimension == 2:
        return as_tensor([[1, 0], [σ[0], h]])
    return as_tensor([[1, 0, 0], [0, 1, 0], [σ[0], σ[1], h]])


def coordinate_transformation_derivative(b, h):
    mesh = ufl.domain.extract_unique_domain(b)
    _, _, ζ = firedrake.SpatialCoordinate(mesh)
    σ = grad(b) + ζ * grad(h)
    if mesh.geometric_dimension == 2:
        return as_tensor([[1, 0], [-σ[0] / h, 1 / h]])
    return as_tensor([[1, 0, 0], [0, 1, 0], [-σ[0] / h, -σ[1] / h, 1 / h]])


def free_energy_rate_terrain_following(**kwargs) -> firedrake.Form:
    u = kwargs["velocity"]
    p = kwargs["pressure"]
    b = kwargs["bed"]
    h = kwargs["thickness"]

    J = coordinate_transformation_derivative(b, h)
    J_inv = coordinate_transformation_derivative_inverse(b, h)

    du = dot(grad(dot(J, u)), J_inv)
    ε = sym(du)
    τ = 2 * μ * ε

    G_cells = (
        0.5 * h * inner(τ, ε) - p * div(h * u) - h * inner(dot(J, f), dot(J, u))
    ) * dx

    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    # TODO: quadruple-check the math
    I = firedrake.Identity(mesh.geometric_dimension)
    G_power = -inner(jump(τ - p * I, dot(J, n)), jump(dot(J, u))) * dS

    α = Constant(kwargs["penalty"])
    γ = firedrake.CellSize(mesh)
    G_penalty = α * μ / (2 * γ) * inner(jump(dot(J, u)), jump(dot(J, u))) * dS

    return G_cells + G_power + G_penalty
