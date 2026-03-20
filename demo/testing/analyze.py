import argparse
import numpy as np
import firedrake
from firedrake import norm

parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("file2")
parser.add_argument("--norm", choices=["L2", "H1", "Hdiv"], default="L2")
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
