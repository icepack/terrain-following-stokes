import pytest
import numpy as np
from numpy import pi as π
import firedrake
from firedrake import assemble, Constant, inner, dx, ds_v, ds_b
from zetastokes import cartesian, terrain_following


lx = 5.0
constants = {"viscosity": 1.0, "gravity": 9.81}
fcparams = {"quadrature_degree": 8}
solver_parameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
}


def topography(x):
    b_0 = Constant(0.0)
    δb = Constant(1 / 8)
    k_b = Constant(3)
    δx_b = Constant(1 / 7)
    b = b_0 + δb * firedrake.cos(2 * π * k_b * (x - δx_b))

    s_0 = Constant(1.0)
    δs = Constant(1 / 8)
    k_s = Constant(5)
    δx_s = Constant(3 / 2)
    s = s_0 + δs * firedrake.cos(2 * π * k_s * (x - δx_s))

    return b, s - b


@pytest.mark.parametrize("element", ["CG", "DQ"])
def test_cartesian(element):
    nx = 60
    nz = int(nx / lx)
    interval = firedrake.UnitIntervalMesh(nx)
    initial_mesh = firedrake.ExtrudedMesh(interval, nz)
    ξ, ζ = firedrake.SpatialCoordinate(initial_mesh)
    b, h = topography(ξ)
    Lx = Constant(lx)
    expr = firedrake.as_vector((Lx * ξ, b + h * ζ))
    Vc = initial_mesh.coordinates.function_space()
    X = firedrake.Function(Vc).interpolate(expr)
    mesh = firedrake.Mesh(X)

    u_elt = firedrake.FiniteElement(element, "quadrilateral", 2)
    u_element = firedrake.VectorElement(u_elt)
    p_element = firedrake.FiniteElement(element, "quadrilateral", 1)
    V = firedrake.FunctionSpace(mesh, u_element)
    Q = firedrake.FunctionSpace(mesh, p_element)
    Z = V * Q
    z = firedrake.Function(Z)

    u, p = firedrake.split(z)
    fields = {"velocity": u, "pressure": p}
    boundary_data = {"dirichlet_ids": ("bottom",), "robin_ids": (1, 2)}
    G = cartesian.free_energy_rate(**fields, **constants, **boundary_data)

    params = {
        "form_compiler_parameters": fcparams,
        "solver_parameters": solver_parameters
    }
    if element == "CG":
        bc = firedrake.DirichletBC(Z.sub(0), 0, boundary_data["dirichlet_ids"])
        params["bcs"] = bc

    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, **params)

    H = firedrake.derivative(F, z)
    free_energy_rate = assemble(firedrake.energy_norm(H, z))
    assert free_energy_rate > 0

    u, p = z.subfunctions
    area = assemble(Constant(1) * dx(mesh))
    u_rms = np.sqrt(assemble(inner(u, u) * dx) / area)

    robin_ids = boundary_data["robin_ids"]
    length = assemble(Constant(1) * ds_v(domain=mesh, subdomain_id=robin_ids))
    n = firedrake.FacetNormal(mesh)
    normal_flow = assemble(inner(u, n) * ds_v(robin_ids)) / length
    assert normal_flow / u_rms < 1 / nx


@pytest.mark.parametrize("element", ["CG", "DQ"])
def test_terrain_following(element):
    nx = 60
    nz = int(nx / lx)
    interval = firedrake.IntervalMesh(nx, lx)
    mesh = firedrake.ExtrudedMesh(interval, nz)

    Lx = Constant(lx)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    b, h = topography(x / Lx)

    u_elt = firedrake.FiniteElement(element, "quadrilateral", 2)
    u_element = firedrake.VectorElement(u_elt)
    p_element = firedrake.FiniteElement(element, "quadrilateral", 1)
    V = firedrake.FunctionSpace(mesh, u_element)
    Q = firedrake.FunctionSpace(mesh, p_element)
    Z = V * Q
    z = firedrake.Function(Z)

    u, p = firedrake.split(z)
    fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h}
    boundary_data = {"dirichlet_ids": ("bottom",), "robin_ids": (1, 2)}
    G = terrain_following.free_energy_rate(**fields, **constants, **boundary_data)
    F = firedrake.derivative(G, z)

    params = {
        "form_compiler_parameters": fcparams,
        "solver_parameters": solver_parameters
    }
    if element == "CG":
        bc = firedrake.DirichletBC(Z.sub(0), 0, boundary_data["dirichlet_ids"])
        params["bcs"] = bc

    firedrake.solve(F == 0, z, **params)

    H = firedrake.derivative(F, z)
    kw = {"form_compiler_parameters": fcparams}
    free_energy_rate = assemble(firedrake.energy_norm(H, z), **kw)
    assert free_energy_rate > 0

    u, p = z.subfunctions
    area = assemble(Constant(1) * dx(mesh))
    u_rms = np.sqrt(assemble(inner(u, u) * dx) / area)

    robin_ids = boundary_data["robin_ids"]
    length = assemble(Constant(1) * ds_v(domain=mesh, subdomain_id=robin_ids))
    n = firedrake.FacetNormal(mesh)
    normal_flow = assemble(inner(u, n) * ds_v(robin_ids)) / length
    assert normal_flow / u_rms < 1 / nx

