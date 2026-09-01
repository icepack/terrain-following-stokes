import numpy as np
import tqdm
import firedrake
from firedrake import as_vector, Constant, dot, inner, ds_b, ds_t
import irksome
from zetastokes import cartesian as xyz, terrain_following as tfc


constants = {"viscosity": 1.0, "density": 1.0, "gravity": 9.81, "skin_depth": 1/8}


def thickness(x):
    h_0, h_1 = Constant(1.0), Constant(0.5)
    return (1 - x) * h_0 + x * h_1


def bed(x):
    b_0, b_1 = Constant(0.0), Constant(0.25)
    return (1 - x) * b_0 + x * b_1


def density_cartesian(x, z):
    X = as_vector((x, z))
    ρ_0 = Constant(1.0)
    ρ_1 = Constant(2.0)

    Y = Constant((2/3, 2/3))
    r = Constant(1/8)
    return ρ_0 + (ρ_1 - ρ_0) * firedrake.exp(-inner(X - Y, X - Y) / r**2)


def density_terrain_following(x, ζ):
    b = bed(x)
    h = thickness(x)
    return density_cartesian(x, b + ζ * h)


def make_velocity(mesh, degree=1):
    u_element = firedrake.FiniteElement("DQ", "quadrilateral", degree + 1)
    p_element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    V = firedrake.VectorFunctionSpace(mesh, u_element)
    Q = firedrake.FunctionSpace(mesh, p_element)
    Z = V * Q

    z = firedrake.Function(Z)

    u, p = firedrake.split(z)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    b = bed(x)
    h = thickness(x)

    fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h}
    boundary_data = {"robin_ids": [1, 2, "bottom", "top"]}

    G_stokes = tfc.free_energy_rate(**fields, **constants, **boundary_data)

    μ = Constant(constants["viscosity"])
    λ = Constant(constants["skin_depth"])
    u_b = Constant((1.0, 0.0))
    u_t = Constant((-1.0, 0.0))
    G_forcing = (
        0.5 * μ / λ * inner(u - u_b, u - u_b) * h * ds_b +
        0.5 * μ / λ * inner(u - u_t, u - u_t) * h * ds_t
    )

    G = G_stokes + G_forcing
    F = firedrake.derivative(G, z)
    pparams = {"form_compiler_parameters": {"quadrature_degree": 8 * degree}}
    firedrake.solve(F == 0, z, **pparams)
    return z.subfunctions[0]


def solve_transport_equation_terrain_following(mesh, degree, *, final_time, num_steps):
    u = make_velocity(mesh, degree=1)

    element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    Q = firedrake.FunctionSpace(mesh, element)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    ρ = firedrake.Function(Q).interpolate(density_terrain_following(x, ζ))
    h = thickness(x)

    F = tfc.density_equation(density=ρ, thickness=h, velocity=u, frame=False)
    t = Constant(0.0)
    dt = Constant(final_time / num_steps)
    method = irksome.BackwardEuler()
    ρs = [ρ.copy(deepcopy=True)]
    solver = irksome.TimeStepper(F, method, t, dt, ρ)
    for step in tqdm.trange(num_steps):
        solver.advance()
        ρs.append(ρ.copy(deepcopy=True))

    return ρs


def solve_transport_equation_cartesian(mesh, degree, *, final_time, num_steps):
    u_tfc = make_velocity(mesh, degree=1)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    b = bed(x)
    h = thickness(x)
    J, _ = tfc.coordinate_transformation_derivatives(b, h)

    u_tfc2xyz = firedrake.Function(u_tfc.function_space())
    u_tfc2xyz.interpolate(dot(J, u_tfc))

    Vc = mesh.coordinates.function_space()
    X = firedrake.Function(Vc).interpolate(as_vector((x, b + ζ * h)))
    mesh_xyz = firedrake.Mesh(X)

    V = firedrake.FunctionSpace(mesh_xyz, u_tfc.ufl_element())
    u = firedrake.Function(V)
    u.dat.data[:] = u_tfc2xyz.dat.data_ro[:]

    element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    Q = firedrake.FunctionSpace(mesh_xyz, element)
    x, z = firedrake.SpatialCoordinate(mesh_xyz)
    ρ = firedrake.Function(Q).interpolate(density_cartesian(x, z))

    F = xyz.density_equation(density=ρ, velocity=u, frame=False)
    t = Constant(0.0)
    dt = Constant(final_time / num_steps)
    method = irksome.BackwardEuler()
    ρs = [ρ.copy(deepcopy=True)]
    solver = irksome.TimeStepper(F, method, t, dt, ρ)
    for step in tqdm.trange(num_steps):
        solver.advance()
        ρs.append(ρ.copy(deepcopy=True))

    Q_tfc = firedrake.FunctionSpace(mesh, element)

    def transfer(ρ):
        ρ_tfc = firedrake.Function(Q_tfc)
        ρ_tfc.dat.data[:] = ρ.dat.data_ro[:]
        return ρ_tfc

    return [transfer(ρ) for ρ in ρs]


def make_mesh(n):
    interval = firedrake.UnitIntervalMesh(n)
    return firedrake.ExtrudedMesh(interval, n)


def test_density_advection():
    ns = [8, 16, 24, 32, 48, 64]
    final_time = 8.0
    u_max = 1.0
    degree = 1  # TODO: make adjustable

    meshes = [make_mesh(n) for n in ns]
    num_steps = np.array([int(n * u_max * final_time) for n in ns])

    ρss_tfc = [
        solve_transport_equation_terrain_following(
            mesh, degree, final_time=final_time, num_steps=num,
        )
        for mesh, num in zip(meshes, num_steps)
    ]

    ρss_xyz = [
        solve_transport_equation_cartesian(
            mesh, degree, final_time=final_time, num_steps=num,
        )
        for mesh, num in zip(meshes, num_steps)
    ]

    errors = np.zeros_like(num_steps, dtype=float)
    for index, (ρs_xyz, ρs_tfc) in enumerate(zip(ρss_xyz, ρss_tfc)):
        δρ = ρs_xyz[-1] - ρs_tfc[-1]
        errors[index] = firedrake.norm(δρ)

    slope, intercept = np.polyfit(np.log(1 / num_steps), np.log(errors), 1)
    print(f"log|ρ_xyz - ρ_tfc| ~ {intercept:.2f} + {slope:.2f} ⋅ δx")
    assert slope > 0.9
