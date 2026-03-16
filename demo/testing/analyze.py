import argparse
import numpy as np
import firedrake
from firedrake import norm

parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("file2")
parser.add_argument("--norm", choices=["L2", "H1", "Hdiv"])
args = parser.parse_args()

with firedrake.CheckpointFile(args.file1, "r") as chk:
    nxs = chk.h5pyfile.attrs["nxs"]
    us1, ps1 = [], []
    for nx in nxs:
        mesh = chk.load_mesh(name=f"domain_rect_{nx}")
        u = chk.load_function(mesh, name=f"u_{nx}")
        p = chk.load_function(mesh, name=f"p_{nx}")
        us1.append(u)
        ps1.append(p)


with firedrake.CheckpointFile(args.file2, "r") as chk:
    nxs_ = chk.h5pyfile.attrs["nxs"]
    assert np.array_equal(nxs, nxs_)
    us2, ps2 = [], []
    for nx in nxs:
        mesh = chk.load_mesh(name=f"domain_rect_{nx}")
        u = chk.load_function(mesh, name=f"u_{nx}")
        p = chk.load_function(mesh, name=f"p_{nx}")
        us2.append(u)
        ps2.append(p)


def transfer(u, v):
    return firedrake.Function(u.function_space()).interpolate(v)


errors = [
    norm(u1 - transfer(u1, u2), args.norm) / norm(u1, args.norm)
    for u1, u2 in zip(us1, us2)
]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.scatter(1 / np.array(nxs), errors, color="tab:blue")
ax.plot(1 / np.array(nxs), errors, color="tab:blue")
plt.show()

fig, ax = plt.subplots()
ax.set_aspect("equal")
firedrake.quiver(us1[1], axes=ax)
plt.show()
