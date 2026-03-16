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

    g = Constant(kwargs["gravity"])
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

    # TODO: estimate this based on element degrees and mesh regularity
    α = Constant(kwargs.get("penalty", 100.0))
    γ = avg(firedrake.CellSize(mesh))
    G_penalty = α * μ / (2 * γ) * inner(jump(u), jump(u)) * dS

    # TODO: add non-zero boundary velocity
    u_Γ = Constant((0,) * mesh.geometric_dimension)
    G_boundary_power = -inner(τ - p * I, sym(outer(u - u_Γ, n))) * ds
    G_boundary_penalty = α * μ / (2 * γ) * inner(u - u_Γ, u - u_Γ) * ds

    return G_cells + G_power + G_penalty + G_boundary_power + G_boundary_penalty


def coordinate_transformation_derivative(b, h):
    mesh = ufl.domain.extract_unique_domain(b)
    d = mesh.geometric_dimension
    ζ = firedrake.SpatialCoordinate(mesh)[d - 1]
    σ = grad(b) + ζ * grad(h)
    if d == 2:
        return as_tensor([[1, 0], [σ[0], h]])
    return as_tensor([[1, 0, 0], [0, 1, 0], [σ[0], σ[1], h]])


def coordinate_transformation_derivative_inverse(b, h):
    mesh = ufl.domain.extract_unique_domain(b)
    d = mesh.geometric_dimension
    ζ = firedrake.SpatialCoordinate(mesh)[d - 1]
    σ = grad(b) + ζ * grad(h)
    if d == 2:
        return as_tensor([[1, 0], [-σ[0] / h, 1 / h]])
    return as_tensor([[1, 0, 0], [0, 1, 0], [-σ[0] / h, -σ[1] / h, 1 / h]])


def free_energy_rate_terrain_following(**kwargs) -> firedrake.Form:
    u = kwargs["velocity"]
    p = kwargs["pressure"]
    b = kwargs["bed"]
    h = kwargs["thickness"]

    J = coordinate_transformation_derivative(b, h)
    J_inv = coordinate_transformation_derivative_inverse(b, h)

    μ = Constant(kwargs["viscosity"])
    du = dot(grad(dot(J, u)), J_inv)
    ε = sym(du)
    τ = 2 * μ * ε

    g = Constant(kwargs["gravity"])
    mesh = ufl.domain.extract_unique_domain(u)
    if mesh.geometric_dimension == 2:
        f = Constant((0, -g))
    elif mesh.geometric_dimension == 3:
        f = Constant((0, 0, -g))

    return (0.5 * h * inner(τ, ε) - p * div(h * u) - h * inner(f, dot(J, u))) * dx


def free_energy_rate_terrain_following_dg(**kwargs) -> firedrake.Form:
    G_cells = free_energy_rate_terrain_following(**kwargs)

    u = kwargs["velocity"]
    p = kwargs["pressure"]
    b = kwargs["bed"]
    h = kwargs["thickness"]

    J = coordinate_transformation_derivative(b, h)
    J_inv = coordinate_transformation_derivative_inverse(b, h)

    μ = Constant(kwargs["viscosity"])
    du = dot(grad(dot(J, u)), J_inv)
    ε = sym(du)
    τ = 2 * μ * ε
    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    from firedrake import dS_h, dS_v, ds_tb, ds_v

    # TODO: quadruple-check the math
    I = firedrake.Identity(mesh.geometric_dimension)
    #G_power = -inner(avg(τ - p * I), u_n("+") + u_n("-")) * h * dS
    u_n = outer(dot(J, u), dot(n, J_inv))
    g_power = (-inner(avg(τ), u_n("+") + u_n("-")) + avg(p) * jump(u, n))
    G_power = g_power * dS_h + g_power * h * dS_v

    α = Constant(kwargs["penalty"])
    γ = firedrake.CellSize(mesh)
    g_penalty = α * μ / (2 * avg(γ)) * inner(jump(dot(J, u)), jump(dot(J, u)))
    G_penalty = g_penalty * dS_h + g_penalty * h * dS_v

    u_Γ = Constant((0,) * mesh.geometric_dimension)
    g_boundary_power = (-inner(τ, u_n) + p * inner(u, n))
    G_boundary_power = g_boundary_power * ds_tb + g_boundary_power * h * ds_v

    g_boundary_penalty = α * μ / (2 * γ) * inner(dot(J, u - u_Γ), dot(J, u - u_Γ))
    G_boundary_penalty = g_boundary_penalty * ds_tb + g_boundary_penalty * h * ds_v

    return G_cells + G_power + G_penalty + G_boundary_power + G_boundary_penalty
