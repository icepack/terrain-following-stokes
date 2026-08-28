import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import firedrake
from firedrake import Constant

nx, nz = 8, 4
interval = firedrake.UnitIntervalMesh(nx)
tfc_mesh = firedrake.ExtrudedMesh(interval, nz)

x, ζ = firedrake.SpatialCoordinate(tfc_mesh)
δb = Constant(0.2)
b_expr = δb * (1 - firedrake.cos(π * x))
s_0, δs_0, δs_1 = Constant(1.0), Constant(0.2), Constant(0.0625)
s_expr = s_0 - δs_0 * x + δs_1 * firedrake.sin(2 * π * x)
z_expr = (1 - ζ) * b_expr + ζ * s_expr
expr = firedrake.as_vector((x, z_expr))

Vc = tfc_mesh.coordinates.function_space()
X = firedrake.Function(Vc).interpolate(expr)
xyz_mesh = firedrake.Mesh(X)

fig, axes = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6.4, 3.2)
)
fig.subplots_adjust(wspace=0.32)   # open up the gutter for the arrow
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

axes[0].set_ylabel(r"$\zeta$", rotation=0, fontsize=16)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
axes[1].set_ylabel(r"$z$", rotation=0, fontsize=16)

axes[0].set_title("Terrain-following\ncoordinates")
axes[1].set_title("Cartesian\ncoordinates")

kw = {"boundary_kw": {"colors": 4 * ["black"]}}

firedrake.triplot(tfc_mesh, axes=axes[0], **kw)
firedrake.triplot(xyz_mesh, axes=axes[1], **kw)

# The axes have equal aspect, so their *drawn* extents are smaller than what
# `get_position` reports; we have to draw once and query the real window extents.
fig.canvas.draw()
inv = fig.transFigure.inverted()
bbox_l = axes[0].get_window_extent().transformed(inv)
bbox_r = axes[1].get_window_extent().transformed(inv)

gap = bbox_r.x0 - bbox_l.x1
pad = 0.12 * gap                       # leave a little air on each side
x0, x1 = bbox_l.x1 + pad, bbox_r.x0 - pad
y = 0.5 * (bbox_l.y0 + bbox_l.y1)      # vertically centered on the panels
arrow = FancyArrowPatch(
    (x0, y),
    (x1, y),
    transform=fig.transFigure,
    arrowstyle="-|>",
    mutation_scale=20,
    linewidth=1.5,
    color="black",
    clip_on=False,
)
fig.add_artist(arrow)

fig.savefig("tfc-vs-xyz.png", dpi=300, bbox_inches="tight")
