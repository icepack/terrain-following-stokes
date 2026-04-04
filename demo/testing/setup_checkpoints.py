r"""Make a checkpoint file containing meshes for the terrain-following and
Cartesian solutions, which will be written into this file at a later stage"""

import argparse
import firedrake
import topography


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography")
    args = parser.parse_args()

    nxs = [16, 20, 24, 32, 48, 64, 72, 84, 96, 108, 128]
    # TODO: stop hard-coding this so we can test sensitivity to aspect ratio
    lx = 5.0
    tfc_meshes = [
        firedrake.ExtrudedMesh(
            firedrake.IntervalMesh(nx, lx, name=f"ival_{nx}"), nx, name=f"rect_{nx}"
        ) for nx in nxs
    ]

    topo_fn = getattr(topography, args.topography)
    xyz_meshes = []
    for initial_mesh in tfc_meshes:
        x, ζ = firedrake.SpatialCoordinate(initial_mesh)
        b, h = topo_fn(x / lx)
        Vc = initial_mesh.coordinates.function_space()
        z = b + h * ζ
        expr = firedrake.as_vector((x, z))
        X = firedrake.Function(Vc).interpolate(expr)
        mesh = firedrake.Mesh(X, name=f"domain_{initial_mesh.name}")
        xyz_meshes.append(mesh)

    with firedrake.CheckpointFile(f"stokes-{args.topography}.h5", "w") as chk:
        chk.h5pyfile.attrs["nxs"] = nxs
        chk.h5pyfile.attrs["topography"] = args.topography
        for meshes in (xyz_meshes, tfc_meshes):
            for mesh in meshes:
                chk.save_mesh(mesh)
