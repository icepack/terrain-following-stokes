import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake
from firedrake import grad, as_vector


with firedrake.CheckpointFile("perlin.h5", "r") as chk:
    num_steps = chk.h5pyfile.attrs["num_steps"]
    mesh = chk.load_mesh(name="domain")
    b = chk.load_function(mesh, name="bed")
    zs = [
        chk.load_function(mesh, name="solution", idx=idx)
        for idx in tqdm.trange(num_steps)
    ]


xs = mesh._base_mesh.coordinates.dat.data_ro
h_0 = zs[0].subfunctions[2]
h_T = zs[-1].subfunctions[2]

B = b.dat.data_ro
S_0 = B + h_0.dat.data_ro
S_T = B + h_T.dat.data_ro

# The velocity lives on the slab mesh with coordinates (x, ζ), 0 ≤ ζ ≤ 1, and
# its components are with respect to that frame. The physical position is
# (x, b + ζh) and the Cartesian velocity is J⋅u, where J is the Jacobian of
# the map from the slab to the physical domain. We sample at points in the
# interior of cells (horizontally at every `stride`-th cell midpoint and
# vertically at the middle of some of the layers) so that the discontinuous
# fields and the gradients of b and h are unambiguous there.
nx = len(xs) - 1
nz = mesh.layers - 1
stride = 20
δx = xs[1] - xs[0]
X_ref = xs[0] + δx * (np.arange(stride // 2, nx, stride) + 0.5)
ζ_ref = (np.arange(1, nz, 2) + 0.5) / nz
points = np.array([(x, ζ) for x in X_ref for ζ in ζ_ref])
point_cloud = firedrake.VertexOnlyMesh(mesh, points, redundant=True)
W = firedrake.VectorFunctionSpace(point_cloud, "DG", 0)

V = zs[0].subfunctions[0].function_space()
H = zs[0].subfunctions[2].function_space()
u, h = firedrake.Function(V), firedrake.Function(H)
x, ζ = firedrake.SpatialCoordinate(mesh)
σ = grad(b)[0] + ζ * grad(h)[0]
position_expr = as_vector([x, b + ζ * h])
velocity_expr = as_vector([u[0], σ * u[0] + h * u[1]])
position = firedrake.Function(W)
velocity = firedrake.Function(W)


def sample(z):
    u.assign(z.subfunctions[0])
    h.assign(z.subfunctions[2])
    position.interpolate(position_expr)
    velocity.interpolate(velocity_expr)
    return position.dat.data_ro.copy(), velocity.dat.data_ro.copy()


samples = [sample(z) for z in tqdm.tqdm(zs)]

# Draw each arrow as the displacement over `arrow_time` years in the plot
# coordinates (km horizontally, m vertically), so the arrows point along the
# flow even with the large vertical exaggeration.
speed_max = max(np.abs(w[:, 0]).max() for _, w in samples)
arrow_time = 0.75 * stride * δx / speed_max


def arrow_components(w):
    return arrow_time * w[:, 0] / 1e3, arrow_time * w[:, 1]


fig, ax = plt.subplots()
ax.set_title("Perlin glacier evolution")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")

ax.plot(xs / 1e3, B, color="tab:brown")
line, = ax.plot(xs / 1e3, S_0, color="tab:blue")

P, w = samples[0]
quiver = ax.quiver(
    P[:, 0] / 1e3,
    P[:, 1],
    *arrow_components(w),
    angles="xy",
    scale_units="xy",
    scale=1,
    width=2e-3,
    minlength=0,
    color="black",
)

def animate(index):
    h = zs[index].subfunctions[2]
    S = B + h.dat.data_ro
    line.set_ydata(S)

    P, w = samples[index]
    quiver.set_offsets(np.column_stack((P[:, 0] / 1e3, P[:, 1])))
    quiver.set_UVC(*arrow_components(w))
    return line, quiver


frames = tqdm.trange(len(zs))
animation = FuncAnimation(fig, animate, frames, interval=1e3/24, blit=True)
animation.save("perlin-glacier.mp4")
