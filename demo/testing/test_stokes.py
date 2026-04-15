import argparse
import itertools
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
    elif basis in ["hdiv", "bdmcf-dpc"]:
        bdm1 = firedrake.FiniteElement("BDMCF", "quadrilateral", 1)
        dpc0 = firedrake.FiniteElement("DPC", "quadrilateral", 0)
        return bdm1, dpc0
    else:
        raise ValueError("`basis` must be either `cg` or `hdiv`!")


def solve(fn_space, bed, thickness, free_energy_rate_fn):
    z = firedrake.Function(fn_space)
    u, p = firedrake.split(z)
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


def solve_sequence(
    meshes, basis, topo_fn, free_energy_rate_fn, topo_degree, verbose=True
):
    u_element, p_element = make_elements(basis)
    solutions = []
    for mesh in meshes:
        V = firedrake.FunctionSpace(mesh, u_element)
        Q = firedrake.FunctionSpace(mesh, p_element)
        Z = V * Q

        lx = Constant(mesh.coordinates.dat.data_ro[:, 0].max())
        b, h = topo_fn(firedrake.SpatialCoordinate(mesh)[0] / lx)
        if topo_degree != "inf":
            cg = firedrake.FiniteElement("CG", "interval", topo_degree)
            r = firedrake.FiniteElement("R", "interval", 0)
            topo_element = firedrake.TensorProductElement(cg, r)
            S = firedrake.FunctionSpace(mesh, topo_element)
            b = firedrake.Function(S).interpolate(b)
            h = firedrake.Function(S).interpolate(h)
        u, p = solve(Z, b, h, free_energy_rate_fn)
        solutions.append((u, p))
        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print("")
    return solutions


# TODO: this is real dog ass
def get_free_energy_rate_function(coordinates, basis, topo_degree):
    match (coordinates, basis, topo_degree):
        case ("xyz", "cg", _):
            return stokes.free_energy_rate_cartesian
        case ("xyz", "hdiv", _):
            return stokes.free_energy_rate_cartesian_dg
        case ("tfc", "cg", "inf"):
            return stokes.free_energy_rate_terrain_following
        case ("tfc", "cg", _):
            return stokes.free_energy_rate_terrain_following_dg
        case ("tfc", "hdiv", _):
            return stokes.free_energy_rate_terrain_following_dg


parser = argparse.ArgumentParser()
parser.add_argument("--filename")
args=  parser.parse_args()

configs = [
    ("xyz", "cg", "inf"),
    ("xyz", "hdiv", "inf"),
    ("tfc", "cg", "inf"),
    ("tfc", "cg", 1),
    ("tfc", "cg", 2),
    ("tfc", "hdiv", "inf"),
    ("tfc", "hdiv", 1),
    ("tfc", "hdiv", 2),
]

for (coords, basis, degree) in configs:
    print(f"{coords} | {basis:4} | {degree}")
    with firedrake.CheckpointFile(args.filename, "r") as chk:
        nxs = chk.h5pyfile.attrs["nxs"]
        topo_name = chk.h5pyfile.attrs["topography"]
        prefix = ("domain_" if coords == "xyz" else "") + "rect"
        meshes = [chk.load_mesh(f"{prefix}_{nx}") for nx in nxs]

    topo_fn = getattr(topography, topo_name)
    G = get_free_energy_rate_function(coords, basis, degree)
    solutions = solve_sequence(meshes, basis, topo_fn, G, degree)

    crd = coords + ("" if coords == "xyz" else f"_{degree}")
    info = f"{crd}_{basis}"
    with firedrake.CheckpointFile(args.filename, "a") as chk:
        for (u, p), nx in zip(solutions, nxs):
            chk.save_function(u, name=f"u_{info}_{nx}")
            chk.save_function(p, name=f"p_{info}_{nx}")
