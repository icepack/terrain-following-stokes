import argparse
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="rayleigh-taylor.h5")
parser.add_argument("--output", type=str, default="rayleigh-taylor-thickness.png")
parser.add_argument("--stride", type=int, default=2, help="plot every n-th step")
parser.add_argument("--max-time", type=float, default=3.0, help="last time to plot (Ma)")
parser.add_argument("--elevation", type=float, default=30.0)
parser.add_argument("--azimuth", type=float, default=-75.0)
args = parser.parse_args()


with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh(name="domain")
    num_steps = chk.h5pyfile.attrs["num_steps"]
    final_time = chk.h5pyfile.attrs["final_time"]
    timestep = final_time / num_steps
    last_step = min(num_steps, int(round(1e3 * args.max_time / timestep)))
    steps = list(range(0, last_step + 1, args.stride))
    if steps[-1] != last_step:
        steps.append(last_step)
    hs = [
        chk.load_function(mesh, name="solution", idx=step).subfunctions[3]
        for step in steps
    ]

# The thickness is CG1 in x and constant in the vertical, so it has one degree
# of freedom per column. Sort them by x to draw each one as a line.
x = firedrake.SpatialCoordinate(mesh)[0]
xs = firedrake.Function(hs[0].function_space()).interpolate(x).dat.data_ro
order = np.argsort(xs)
xs = xs[order]
times = np.array(steps) * timestep / 1e3  # Ma

heights = [h.dat.data_ro[order] for h in hs]
z_min = min(z.min() for z in heights)
z_max = max(z.max() for z in heights)
z_floor = z_min - 0.05 * (z_max - z_min)

# Draw as a waterfall plot: each curve sits on an opaque fill that hides the
# curves behind it. Matplotlib's 3D depth sorting can't get the order of
# overlapping lines right, so we set it by hand. The camera looks from the t = 0
# side, so earlier slices are nearer and go on top.
fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
ax = fig.add_subplot(projection="3d", computed_zorder=False)
for index, (t, z) in enumerate(zip(times, heights)):
    ts = np.full_like(xs, t)
    zorder = 2 * (len(times) - index)
    verts = [(xs[0], t, z_floor), *zip(xs, ts, z), (xs[-1], t, z_floor)]
    fill = Poly3DCollection([verts], facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_collection3d(fill)
    ax.plot(xs, ts, z, color="black", linewidth=0.8, zorder=zorder + 1)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(times[0], times[-1])
ax.set_zlim(z_floor, z_max)

ax.set_xlabel("x (km)")
ax.set_ylabel("t (Ma)")
ax.set_zlabel("Thickness (km)", labelpad=12)
ax.zaxis.get_major_formatter().set_useOffset(False)
# The fills are white, so hide the panes and grid that they would partly cover.
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("none")
ax.view_init(elev=args.elevation, azim=args.azimuth)
fig.savefig(args.output, dpi=300, bbox_inches="tight")
