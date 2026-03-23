import argparse
import numpy as np
import ufl
import firedrake
from firedrake import Constant, dot
import stokes
import topography


coefficients = {
    "viscosity": 1.0,
    "gravity": 9.81,
}


def make_elements(basis):
    if basis in ["cg", "taylor-hood"]:
        cg2 = firedrake.FiniteElement("CG", "quadrilateral", 2)
        cg1 = firedrake.FiniteElement("CG", "quadrilateral", 1)
        return firedrake.VectorElement(cg2), cg1
    elif basis in ["hdiv", "bdmcf-dg"]:
        bdm1 = firedrake.FiniteElement("BDMCF", "quadrilateral", 1)
        dg0 = firedrake.FiniteElement("DQ", "quadrilateral", 0)
        return bdm1, dg0
    else:
        raise ValueError("`basis` must be either `cg` or `hdiv`!")


def solve(fn_space, bed, thickness, terrain_following, free_energy_rate_fn):
    z = firedrake.Function(fn_space)
    u, p = firedrake.split(z)
    # TODO: fix this up for when we use Functions, not expressions
    fields = {"velocity": u, "pressure": p, "bed": bed, "thickness": thickness}

    dirichlet_ids = [1, 2, "bottom"]
    bcs = firedrake.DirichletBC(fn_space.sub(0), 0, dirichlet_ids)

    boundary_data = {"dirichlet_ids": dirichlet_ids}
    G = free_energy_rate_fn(**fields, **coefficients, **boundary_data)
    F = firedrake.derivative(G, z)
    params = {
        "form_compiler_parameters": {"quadrature_degree": 10},  # FIXME
        "solver_parameters": {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    firedrake.solve(F == 0, z, bcs=bcs, **params)
    return z.subfunctions


def main(
    meshes,
    basis,
    topo_fn,
    terrain_following,
    free_energy_rate_fn,
    topo_degree=None,
    verbose=True
):
    u_element, p_element = make_elements(basis)

    solutions = []
    for initial_mesh in meshes:
        lx = Constant(initial_mesh.coordinates.dat.data_ro[:, 0].max())
        x, ζ = firedrake.SpatialCoordinate(initial_mesh)
        b, h = topo_fn(x / lx)
        fn_space = initial_mesh.coordinates.function_space()
        z = ζ if terrain_following else b + h * ζ
        expr = firedrake.as_vector((x, z))
        X = firedrake.Function(fn_space).interpolate(expr)
        mesh = firedrake.Mesh(X, name=f"domain_{initial_mesh.name}")
        V = firedrake.FunctionSpace(mesh, u_element)
        Q = firedrake.FunctionSpace(mesh, p_element)
        Z = V * Q

        b, h = topo_fn(firedrake.SpatialCoordinate(mesh)[0] / lx)
        if topo_degree != None:
            cg = firedrake.FiniteElement("CG", "interval", topo_degree)
            r = firedrake.FiniteElement("R", "interval", 0)
            topo_element = firedrake.TensorProductElement(cg, r)
            S = firedrake.FunctionSpace(mesh, topo_element)
            b = firedrake.Function(S).interpolate(b)
            h = firedrake.Function(S).interpolate(h)
        u, p = solve(Z, b, h, terrain_following, free_energy_rate_fn)
        solutions.append((u, p))
        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print("")
    return solutions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography", choices=["linear", "wavy"])
    parser.add_argument("--basis", choices=["cg", "hdiv"])
    parser.add_argument("--coordinates", choices=["cartesian", "terrain-following"])
    parser.add_argument("--topo-degree", type=int)
    args = parser.parse_args()

    nxs = [16, 20, 24, 32, 48, 64, 72, 84, 96, 108, 128]
    lx = 5.0
    meshes = [
        firedrake.ExtrudedMesh(
            firedrake.IntervalMesh(nx, lx, name=f"ival_{nx}"), nx, name=f"rect_{nx}"
        ) for nx in nxs
    ]

    topo_fns = {"linear": topography.linear, "wavy": topography.wavy}
    main_args = [meshes, args.basis, topo_fns[args.topography]]
    tf = args.coordinates == "terrain-following"
    match (args.coordinates, args.basis):
        case ("cartesian", "cg"):
            G_fn = stokes.free_energy_rate_cartesian
        case ("cartesian", "hdiv"):
            G_fn = stokes.free_energy_rate_cartesian_dg
        case ("terrain-following", "cg"):
            G_fn = stokes.free_energy_rate_terrain_following
        case ("terrain-following", "hdiv"):
            G_fn = stokes.free_energy_rate_terrain_following_dg
            main_args[1] = "cg"
    solutions = main(*main_args, tf, G_fn, args.topo_degree)

    degree = str(args.topo_degree or "inf")
    filename = f"stokes-{args.topography}-{degree}-{args.basis}-{args.coordinates}.h5"
    with firedrake.CheckpointFile(filename, "w") as chk:
        chk.h5pyfile.attrs["nxs"] = nxs
        chk.h5pyfile.attrs["topography"] = args.topography
        chk.h5pyfile.attrs["coordinates"] = args.coordinates
        for (u, p), nx in zip(solutions, nxs):
            chk.save_function(u, name=f"u_{nx}")
            chk.save_function(p, name=f"p_{nx}")
