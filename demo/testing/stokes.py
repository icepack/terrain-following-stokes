import firedrake
from firedrake import (
    Constant, as_tensor, inner, dot, sym, grad, div, dx, ds, dS, avg, jump
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
    G_conforming = free_energy_rate_cartesian(**kwargs)

    u = kwargs["velocity"]
    p = kwargs["pressure"]

    μ = Constant(kwargs["viscosity"])
    ε = sym(grad(u))
    τ = 2 * μ * ε

    mesh = ufl.domain.extract_unique_domain(u)
    n = firedrake.FacetNormal(mesh)

    # TODO: quadruple-check the math here
    I = firedrake.Identity(mesh.geometric_dimension)
    G_power = -inner(jump(τ - p * I, n), jump(u)) * dS

    α = Constant(kwargs["penalty"])
    γ = avg(firedrake.CellSize(mesh))
    G_penalty = α * μ / (2 * γ) * inner(jump(u), jump(u)) * dS

    # TODO: add non-zero boundary velocity
    ids = tuple(kwargs["dirichlet_ids"])
    u_Γ = Constant((0,) * mesh.geometric_dimension)
    G_boundary_power = -inner(dot(τ - p * I, n), u - u_Γ) * ds(ids)
    G_boundary_penalty = α * μ / (2 * γ) * inner(u - u_Γ, u - u_Γ) * ds(ids)

    return G_conforming + G_power + G_penalty + G_boundary_power + G_boundary_penalty


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

    I = firedrake.Identity(mesh.geometric_dimension)

    f_x = dot(J, f)
    u_x = dot(J, u)
    G_cells = (0.5 * h * inner(τ, ε) - p * div(h * u) - h * inner(f_x, u_x)) * dx

    # Miracle occurs...

    raise NotImplementedError("Insh'allah I will finish this")
    return G_cells + G_power + G_penalty
