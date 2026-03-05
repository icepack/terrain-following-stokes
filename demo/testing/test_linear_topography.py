import argparse
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import Constant
import stokes



coefficients = {
    "viscosity": 1.0,
    "gravity": 9.81,
}


def make_mesh_and_dirichlet_ids(extruded: bool, nx: int, nz: int):
    if extruded:
        interval = firedrake.UnitIntervalMesh(nx)
        initial_mesh = firedrake.ExtrudedMesh(interval, nz)
        dirichlet_ids = [1, 2, "bottom"]
    else:
        initial_mesh = firedrake.UnitSquareMesh(nx, nz, quadrilateral=True)
        dirichlet_ids = [1, 2, 3]

    x, z = firedrake.SpatialCoordinate(initial_mesh)

    b_0 = Constant(-1)
    b_1 = Constant(-0.5)

    s_0 = Constant(1.0)
    s_1 = Constant(0.5)

    lx = Constant(5.0)

    b = (1 - x) * b_0 + x * b_1
    s = (1 - x) * s_0 + x * s_1
    expr = firedrake.as_vector((lx * x, (1 - z) * b + z * s))
    fn_space = initial_mesh.coordinates.function_space()
    X = firedrake.Function(fn_space).interpolate(expr)
    mesh = firedrake.Mesh(X)

    return mesh, dirichlet_ids


def main(mesh, dirichlet_ids, basis: str) -> firedrake.Function:
    if basis == "cg":
        cg2 = firedrake.FiniteElement("CG", "quadrilateral", 2)
        cg1 = firedrake.FiniteElement("CG", "quadrilateral", 1)
        V = firedrake.VectorFunctionSpace(mesh, cg2)
        Q = firedrake.FunctionSpace(mesh, cg1)
        free_energy_rate = stokes.free_energy_rate_cartesian
    elif basis == "hdiv":
        bdm1 = firedrake.FiniteElement("BDMCF", "quadrilateral", 1)
        dg0 = firedrake.FiniteElement("DQ", "quadrilateral", 0)
        V = firedrake.FunctionSpace(mesh, bdm1)
        Q = firedrake.FunctionSpace(mesh, dg0)
        free_energy_rate = stokes.free_energy_rate_cartesian_dg

        # TODO: figure out what this should be based on mesh regularity
        coefficients["penalty"] = 100.0

    Z = V * Q

    z = firedrake.Function(Z)
    u, p = firedrake.split(z)
    fields = {"velocity": u, "pressure": p}

    bcs = firedrake.DirichletBC(Z.sub(0), 0, dirichlet_ids)

    boundary_data = {"dirichlet_ids": dirichlet_ids}
    G = free_energy_rate(**fields, **coefficients, **boundary_data)
    F = firedrake.derivative(G, z)
    firedrake.solve(F == 0, z, bcs=bcs)

    return z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extruded", action="store_true")
    args = parser.parse_args()

    expmin = 3
    expmax = 8
    num = expmax - expmin + 1
    nxs = 2 ** np.linspace(expmin, expmax, num, dtype=int)
    errors_u = np.zeros(num)
    errors_p = np.zeros(num)

    for index, nx in enumerate(nxs):
        mesh, dirichlet_ids = make_mesh_and_dirichlet_ids(args.extruded, nx, nx)
        z_cg = main(mesh, dirichlet_ids, "cg")
        z_hdiv = main(mesh, dirichlet_ids, "hdiv")
        u_cg, p_cg = z_cg.subfunctions
        u_hdiv, p_hdiv = z_hdiv.subfunctions
        errors_u[index] = firedrake.norm(u_cg - u_hdiv) / firedrake.norm(u_cg)
        errors_p[index] = firedrake.norm(p_cg - p_hdiv) / firedrake.norm(p_cg)
        print(".", end="", flush=True)

    fig, ax = plt.subplots()
    ax.scatter(1 / nxs, errors_u, color="tab:blue")
    ax.plot(1 / nxs, errors_u, color="tab:blue", label="velocity diffs")
    ax.scatter(1 / nxs, errors_p, color="tab:orange")
    ax.plot(1 / nxs, errors_p, color="tab:orange", label="pressure diffs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    plt.show()
