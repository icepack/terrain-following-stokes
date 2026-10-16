import numpy as np
import firedrake
from firedrake import (
    assemble, sqrt, dot, grad, inner, outer, as_vector, Constant, Function,
    ds_b, ds_t, ds_v, dx
)
import ufl
from zetastokes import cartesian
from zetastokes.terrain_following import (
    free_energy_rate,
    coordinate_surface_slope,
    coordinate_transformation_derivatives,
)


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

    σ = coordinate_surface_slope(b=b, h=h)
    J, J_inv = coordinate_transformation_derivatives(b=b, h=h)

    fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h}
    boundary_data = {"robin_ids": [1, 2, "bottom", "top"]}
    G_stokes = free_energy_rate(**fields, **constants, **boundary_data)

    # Form the projection onto the plane tangent to the top/bottom surface
    ν = -σ + Constant((0.0, 1.0))
    I = firedrake.Identity(2)
    Π = I - outer(ν, ν) / (1 + inner(σ, σ))

    u_t = Constant((1.0, 0.0))
    u_b = Constant((-1.0, 0.0))
    μ = Constant(constants["viscosity"])
    λ = Constant(constants["skin_depth"])

    ΠJu_t = dot(Π, dot(J, u - u_t))
    ΠJu_b = dot(Π, dot(J, u - u_b))

    γ = sqrt(1 + inner(σ, σ))

    G_forcing = (
        0.5 * μ / λ * inner(ΠJu_t, ΠJu_t) * γ * ds_t +
        0.5 * μ / λ * inner(ΠJu_b, ΠJu_b) * γ * ds_b
    )
    G = G_stokes + G_forcing

    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, **pparams)
    return z


def physical_mesh(mesh):
    x, ζ = firedrake.SpatialCoordinate(mesh)
    expr = as_vector((x, bed(x) + thickness(x) * ζ))
    X = Function(mesh.coordinates.function_space()).interpolate(expr)
    return firedrake.Mesh(X)


def solve_cartesian(mesh, degree):
    z = setup(mesh, degree)
    u, p = firedrake.split(z)
    x = firedrake.SpatialCoordinate(mesh)[0]

    fields = {"velocity": u, "pressure": p}
    boundary_data = {"robin_ids": [1, 2, "bottom", "top"]}
    G_stokes = cartesian.free_energy_rate(**fields, **constants, **boundary_data)

    n = firedrake.FacetNormal(mesh)
    I = firedrake.Identity(2)
    Π = I - outer(n, n)

    # Cartesian equivalents `J u_t`, `J u_b` of the terrain-following data
    grad_s = grad(bed(x) + thickness(x))[0]
    grad_b = grad(bed(x))[0]
    u_t = as_vector((1.0, grad_s))
    u_b = as_vector((-1.0, -grad_b))
    μ = Constant(constants["viscosity"])
    λ = Constant(constants["skin_depth"])

    Πu_t = dot(Π, u - u_t)
    Πu_b = dot(Π, u - u_b)
    G_forcing = (
        0.5 * μ / λ * inner(Πu_t, Πu_t) * ds_t +
        0.5 * μ / λ * inner(Πu_b, Πu_b) * ds_b
    )
    G = G_stokes + G_forcing

    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, **pparams)
    return z


def tfc_to_xyz(u, mesh_xyz, degree=None):
    mesh = ufl.domain.extract_unique_domain(u)
    x, _ = firedrake.SpatialCoordinate(mesh)
    b, h = bed(x), thickness(x)
    J, _ = coordinate_transformation_derivatives(b=b, h=h)

    # Create the Cartesian velocity, defined on the terrain-following mesh
    if degree is None:
        hdegree = 2
        degree = max(*u.ufl_element().degree()) + hdegree
    element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    V_dg = firedrake.VectorFunctionSpace(mesh, element)
    Ju = Function(V_dg).project(dot(J, u))

    # Transfer the Cartesian velocity to the Cartesian mesh
    V_xyz = firedrake.VectorFunctionSpace(mesh_xyz, element)
    u_xyz = Function(V_xyz)
    u_xyz.dat.data[:] = Ju.dat.data_ro[:]

    return u_xyz


def test_robin_tfc_vs_xyz():
    nxs = np.array([8, 16, 32, 64])
    errors = np.zeros_like(nxs, dtype=float)
    flux_errors = np.zeros_like(nxs, dtype=float)
    for index, nx in enumerate(nxs):
        interval = firedrake.UnitIntervalMesh(nx)
        mesh = firedrake.ExtrudedMesh(interval, nx)
        mesh_xyz = physical_mesh(mesh)
        u, p = solve_terrain_following(mesh, degree=1).subfunctions

        # Measure the misfit between the terrain-following and Cartesian
        # solutions
        u_xyz, p_xyz = solve_cartesian(mesh_xyz, degree=1).subfunctions
        u_xfer = tfc_to_xyz(u, mesh_xyz)
        errors[index] = firedrake.norm(u_xfer - u_xyz) / firedrake.norm(u_xyz)

        # Measure the flux out of the boundaries (which should be 0)
        n = firedrake.FacetNormal(mesh)
        length = assemble(Constant(1) * (ds_b(mesh) + ds_t(mesh) + ds_v(mesh)))
        flux_avg = assemble(abs(inner(u, n)) * (ds_b + ds_t + ds_v)) / length
        area = assemble(Constant(1) * dx(mesh))
        speed_avg = np.sqrt(assemble(inner(u, u) * dx) / area)
        flux_errors[index] = flux_avg / speed_avg

    slope, intercept = np.polyfit(np.log(1 / nxs), np.log(errors), 1)
    print(f"log |u_tfc - u_xyz| / |u_xyz| ~ {intercept:0.2f} + {slope:0.2f} * log(δx)")
    assert slope > 0.9

    slope, intercept = np.polyfit(np.log(1 / nxs), np.log(flux_errors), 1)
    print(f"log ∫|u · n| ds ~= {intercept:0.2f} + {slope:0.2f} * log(δx)")
    assert slope > 0.9
