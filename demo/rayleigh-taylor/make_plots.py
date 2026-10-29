import argparse
import numpy as np
import matplotlib.pyplot as plt
import firedrake


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="rayleigh-taylor.h5")
parser.add_argument("--output", type=str, default="rayleigh-taylor-density.png")
args = parser.parse_args()


with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh(name="domain")
    num_steps = chk.h5pyfile.attrs["num_steps"]
    final_time = chk.h5pyfile.attrs["final_time"]
    zs = [
        chk.load_function(mesh, name="solution", idx=step)
        for step in range(num_steps + 1)
    ]

nrows, ncols = 4, 4
num = nrows * ncols
indices = (num_steps * np.linspace(0.0, 1.0, num)).astype(int)
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, sharex=True, sharey=True, layout="compressed"
)

kw = {"vmin": 3200, "vmax": 3300, "cmap": "Blues"}
for index, ax in zip(indices, axes.flatten()):
    ax.set_axis_off()
    ρ = zs[index].subfunctions[2]
    firedrake.tripcolor(ρ, axes=ax, **kw)

fig.savefig(f"{args.output}", dpi=300, bbox_inches="tight")
