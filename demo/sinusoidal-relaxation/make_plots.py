import numpy as np
import matplotlib.pyplot as plt
import tqdm
import firedrake


with firedrake.CheckpointFile("relaxation.h5", "r") as chk:
    mesh = chk.load_mesh(name="domain")
    num_steps = chk.h5pyfile.attrs["num_steps"]
    zs = [
        chk.load_function(mesh, name="solution", idx=step)
        for step in tqdm.trange(num_steps)
    ]

hs = [z.subfunctions[2].dat.data_ro for z in zs]

fig, ax = plt.subplots(figsize=(6.4, 3.2))
xs = mesh._base_mesh.coordinates.dat.data_ro / 1e3
ax.set_title("Relaxation of sinusoidal surface")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")

bs = np.zeros_like(xs)
ax.plot(xs, hs[0], "--", color="tab:blue", label="Initial surface")
ax.plot(xs, hs[-1], color="tab:blue", label="Final surface")
for h in hs[1::4]:
    ax.plot(xs, h, linewidth=0.5, color="tab:blue", label="_")
ax.legend(loc="upper right")
fig.savefig("relaxation.png", dpi=300, bbox_inches="tight")
