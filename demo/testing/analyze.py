import argparse
import numpy as np
import ufl
import firedrake
from firedrake import norm, Constant, dot
import stokes
import topography


def terrain_following_to_cartesian(u, p, b, h):
    J = stokes.coordinate_transformation_derivative(b, h)
    # TODO: Check if this should really be `project` or possibly also
    # interpolate into a DG space.
    Ju = firedrake.Function(u.function_space()).interpolate(dot(J, u))

    mesh = ufl.domain.extract_unique_domain(u)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    expr = firedrake.as_vector((x, b + h * ζ))
    Vc = mesh.coordinates.function_space()
    X = firedrake.Function(Vc).interpolate(expr)
    cartesian_mesh = firedrake.Mesh(X, name=mesh.name)

    V = firedrake.FunctionSpace(cartesian_mesh, u.ufl_element())
    Q = firedrake.FunctionSpace(cartesian_mesh, p.ufl_element())

    v = firedrake.Function(V)
    v.dat.data[:] = Ju.dat.data_ro[:]
    q = firedrake.Function(Q)
    q.dat.data[:] = p.dat.data_ro[:]

    return v, q


def load_data(filename):
    with firedrake.CheckpointFile(filename, "r") as chk:
        if chk.h5pyfile.attrs["coordinates"] == "terrain-following":
            match chk.h5pyfile.attrs["topography"]:
                case "linear":
                    topo_fn = topography.linear
                case "wavy":
                    topo_fn = topography.wavy

            def conversion_fn(mesh, u, p):
                x = firedrake.SpatialCoordinate(mesh)[0]
                lx = mesh.coordinates.dat.data_ro[:, 0].max()
                b, h = topo_fn(x / Constant(lx))
                return terrain_following_to_cartesian(u, p, b, h)
        else:
            conversion_fn = lambda mesh, u, p: (u, p)

        nxs = chk.h5pyfile.attrs["nxs"]
        us, ps = [], []
        for nx in nxs:
            mesh = chk.load_mesh(name=f"domain_rect_{nx}")
            u = chk.load_function(mesh, name=f"u_{nx}")
            p = chk.load_function(mesh, name=f"p_{nx}")
            u, p = conversion_fn(mesh, u, p)
            us.append(u)
            ps.append(p)

    return nxs, us, ps


parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("file2")
parser.add_argument("--norm", choices=["L2", "H1", "Hdiv"], default="L2")
args = parser.parse_args()

nxs1, us1, ps1 = load_data(args.file1)
nxs2, us2, ps2 = load_data(args.file2)

assert np.array_equal(nxs1, nxs2)
nxs = nxs1


def transfer(u, v):
    return firedrake.Function(u.function_space()).interpolate(v)


velocity_errors = [
    norm(u1 - transfer(u1, u2), args.norm) / norm(u1, args.norm)
    for u1, u2 in zip(us1, us2)
]

pressure_errors = [
    norm(p1 - transfer(p1, p2)) / norm(p1) for p1, p2 in zip(ps1, ps2)
]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
dxs = 1 / np.array(nxs)

ax.scatter(dxs, velocity_errors, color="tab:blue")
ax.plot(dxs, velocity_errors, color="tab:blue", label="Velocity")
ax.scatter(dxs, pressure_errors, color="tab:orange")
ax.plot(dxs, pressure_errors, color="tab:orange", label="Pressure")
ax.legend(loc="lower right")
ax.set_ylabel("Relative difference")
ax.set_xlabel("Mesh spacing")
plt.show()

uslope, uintercept = np.polyfit(np.log(dxs), np.log(velocity_errors), 1)
print(f"Velocity difference ~= {np.exp(uintercept):0.2g} * dx^{uslope:.1f}")
pslope, pintercept = np.polyfit(np.log(dxs), np.log(pressure_errors), 1)
print(f"Pressure difference ~= {np.exp(pintercept):0.2g} * dx^{pslope:.1f}")


fig, ax = plt.subplots()
ax.set_aspect("equal")
firedrake.streamplot(us1[-1], seed=1729, resolution=0.125, axes=ax)
plt.show()
