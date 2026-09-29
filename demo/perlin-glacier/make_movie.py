import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake


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

fig, ax = plt.subplots()
ax.set_title("Perlin glacier evolution")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")

ax.plot(xs / 1e3, B, color="tab:brown")
line, = ax.plot(xs / 1e3, S_0, color="tab:blue")

def animate(z):
    h = z.subfunctions[2]
    S = B + h.dat.data_ro
    line.set_ydata(S)
    return line,

animation = FuncAnimation(fig, animate, tqdm.tqdm(zs), interval=1e3/24, blit=True)
animation.save("perlin-glacier.mp4")
