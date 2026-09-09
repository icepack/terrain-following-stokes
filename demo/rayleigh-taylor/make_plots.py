import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake


with firedrake.CheckpointFile("rayleigh-taylor.h5", "r") as chk:
    mesh = chk.load_mesh(name="domain")
    num_steps = chk.h5pyfile.attrs["num_steps"]
    final_time = chk.h5pyfile.attrs["final_time"]
    zs = [
        chk.load_function(mesh, name="solution", idx=step)
        for step in range(num_steps)
    ]


u, p, ρ, h = zs[0].subfunctions

fig, ax = plt.subplots()
ax.set_axis_off()
kw = {"num_sample_points": 16, "shading": "gouraud", "cmap": "Blues"}
colors = firedrake.tripcolor(ρ, **kw, axes=ax)

fn_plotter = firedrake.FunctionPlotter(mesh, num_sample_points=16)
def animate(z):
    u, p, ρ, h = z.subfunctions
    colors.set_array(fn_plotter(ρ))

animation = FuncAnimation(fig, animate, zs, interval=1e3/30)
animation.save("rayleigh-taylor.mp4")


fig, axes = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6.4, 3.2)
)
for ax in axes:
    ax.set_axis_off()

axes[0].set_title("0 ka")
axes[1].set_title(f"{int(final_time)} ka")

ρ_0 = zs[0].subfunctions[2]
ρ_1 = zs[-1].subfunctions[2]
kw = {"vmin": 3200, "vmax": 3300, "cmap": "Blues"}
firedrake.tripcolor(ρ_0, axes=axes[0], **kw)
firedrake.tripcolor(ρ_1, axes=axes[1], **kw)
fig.savefig("rayleigh-taylor-density.png", dpi=300, bbox_inches="tight")
