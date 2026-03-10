import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import Constant, norm
import stokes


coefficients = {
    "viscosity": 1.0,
    "gravity": 9.81,
    # TODO: stop being lazy about this
    "penalty": 100.0,
}


b_0 = Constant(-1.0)
b_1 = Constant(-0.5)
s_0 = Constant(1.0)
s_1 = Constant(0.5)

lx = Constant(5.0)


def make_mesh(nx, nz, terrain_following):
    interval = firedrake.UnitIntervalMesh(nx)
    initial_mesh = firedrake.ExtrudedMesh(interval, nz)
    x, ζ = firedrake.SpatialCoordinate(initial_mesh)
    b = (1 - x) * b_0 + x * b_1
    s = (1 - x) * s_0 + x * s_1

    fn_space = initial_mesh.coordinates.function_space()
    if terrain_following:
        expr = firedrake.as_vector((lx * x, ζ))
    else:
        expr = firedrake.as_vector((lx * x, (1 - ζ) * b + ζ * s))
    X = firedrake.Function(fn_space).interpolate(expr)
    mesh = firedrake.Mesh(X)

    x = firedrake.SpatialCoordinate(mesh)[0]
    b = (1 - x / lx) * b_0 + x / lx * b_1
    s = (1 - x / lx) * s_0 + x / lx * s_1
    h = s - b

    return mesh, b, h


def main(mesh, b, h, terrain_following) -> firedrake.Function:
    if terrain_following:
        free_energy_rate = stokes.free_energy_rate_terrain_following
    else:
        free_energy_rate = stokes.free_energy_rate_cartesian

    cg2 = firedrake.FiniteElement("CG", "quadrilateral", 2)
    cg1 = firedrake.FiniteElement("CG", "quadrilateral", 1)
    V = firedrake.VectorFunctionSpace(mesh, cg2)
    Q = firedrake.FunctionSpace(mesh, cg1)
    Z = V * Q

    z = firedrake.Function(Z)
    u, p = firedrake.split(z)
    fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h}

    dirichlet_ids = [1, 2, "bottom"]
    bcs = firedrake.DirichletBC(Z.sub(0), 0, dirichlet_ids)
    boundary_data = {"dirichlet_ids": dirichlet_ids}
    G = free_energy_rate(**fields, **coefficients, **boundary_data)
    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, bcs=bcs)

    return z


if __name__ == "__main__":
    expmin = 3
    expmax = 8
    num = expmax - expmin + 1
    nxs = 2 ** np.linspace(expmin, expmax, num, dtype=int)
    errors_u = np.zeros(num)
    errors_p = np.zeros(num)

    for index, nx in enumerate(nxs):
        mesh_xyz, b, h = make_mesh(nx, nx, terrain_following=False)
        z_xyz = main(mesh_xyz, b, h, terrain_following=False)

        mesh_tfc, b, h = make_mesh(nx, nx, terrain_following=True)
        z_tfc = main(mesh_tfc, b, h, terrain_following=True)

        u_xyz, p_xyz = z_xyz.subfunctions
        u_tfc, p_tfc = z_tfc.subfunctions

        # If you have a better idea for how to do this I am all ears
        J = stokes.coordinate_transformation_derivative(b, h)
        v = firedrake.Function(u_tfc.function_space()).interpolate(firedrake.dot(J, u_tfc))
        u_tfc2xyz = firedrake.Function(u_xyz.function_space())
        u_tfc2xyz.dat.data[:] = v.dat.data_ro[:]

        p_tfc2xyz = firedrake.Function(p_xyz.function_space())
        p_tfc2xyz.dat.data[:] = p_tfc.dat.data_ro[:]

        errors_u[index] = norm(u_xyz - u_tfc2xyz, "H1") / norm(u_xyz, "H1")
        errors_p[index] = norm(p_xyz - p_tfc2xyz, "L2") / norm(p_xyz, "L2")
        print(".", end="", flush=True)

    log_dx = np.log2(1 / nxs)
    log_error_u = np.log2(errors_u)
    log_error_p = np.log2(errors_p)
    slope_u, intercept_u = np.polyfit(log_dx, log_error_u, 1)
    slope_p, intercept_p = np.polyfit(log_dx, log_error_p, 1)
    print(f"log(u error) ~= {slope_u:g} * log(dx) {intercept_u:+g}")
    print(f"log(p error) ~= {slope_p:g} * log(dx) {intercept_p:+g}")

    fig, ax = plt.subplots()
    ax.scatter(1 / nxs, errors_u, color="tab:blue")
    ax.plot(1 / nxs, errors_u, color="tab:blue", label="velocity diffs")
    ax.scatter(1 / nxs, errors_p, color="tab:orange")
    ax.plot(1 / nxs, errors_p, color="tab:orange", label="pressure diffs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    plt.show()
