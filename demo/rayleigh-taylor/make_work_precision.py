import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import assemble, dot, inner, dx
from ufl.domain import extract_unique_domain
from zetastokes.terrain_following import coordinate_transformation_derivatives


parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=str, default="work-prec-radau-12.5.h5")
parser.add_argument("--inputs", type=str, nargs="+", required=True)
parser.add_argument("--output", type=str, default="work-precision.png")
parser.add_argument("--convergence-output", type=str, default="convergence.png")
args = parser.parse_args()


def load_final_state(filename):
    with firedrake.CheckpointFile(filename, "r") as chk:
        mesh = chk.load_mesh(name="domain")
        num_steps = chk.h5pyfile.attrs["num_steps"]
        final_time = chk.h5pyfile.attrs["final_time"]
        z = chk.load_function(mesh, name="solution", idx=num_steps)
    return z, num_steps, final_time


# Every run is saved with its own copy of the mesh. They're all identical, so we
# can copy the degrees of freedom onto the reference mesh and compare them there.
def transfer(z, Z):
    coords_src = extract_unique_domain(z).coordinates.dat.data_ro
    coords_dst = mesh.coordinates.dat.data_ro
    if not np.array_equal(coords_src, coords_dst):
        raise ValueError("Meshes differ, can't compare solutions by DOF")

    w = firedrake.Function(Z)
    for dst, src in zip(w.subfunctions, z.subfunctions):
        dst.dat.data[:] = src.dat.data_ro
    return w


# The velocity is stored in terrain-following components, so map it back to
# Cartesian components using each solution's own thickness. Velocity is in
# km / ka = m / a, density in kg / m^3, and thickness in km.
def physical_fields(z, b):
    u, p, ρ, h = firedrake.split(z)
    J, _ = coordinate_transformation_derivatives(b, h)
    return {"velocity": dot(J, u), "density": ρ, "thickness": 1e3 * h}


z_ref, num_steps_ref, final_time = load_final_state(args.reference)
Z = z_ref.function_space()
mesh = extract_unique_domain(z_ref)
b = firedrake.Function(z_ref.subfunctions[3].function_space())
fields_ref = physical_fields(z_ref, b)
area = assemble(firedrake.Constant(1.0) * dx(mesh))

pattern = re.compile(r"work-prec-(?P<method>\w+)-(?P<timestep>[\d.]+)\.h5$")
runs = []
for filename in args.inputs:
    if filename == args.reference:
        continue

    match = pattern.search(filename)
    method, timestep = match["method"], float(match["timestep"])

    z, num_steps, run_final_time = load_final_state(filename)
    if not np.isclose(num_steps * timestep, final_time):
        raise ValueError(f"{filename} stopped at t = {num_steps * timestep} ka")

    fields = physical_fields(transfer(z, Z), b)
    errors = {
        name: np.sqrt(assemble(inner(f - fields_ref[name], f - fields_ref[name]) * dx) / area)
        for name, f in fields.items()
    }

    with open(filename.removesuffix(".h5") + ".time", "r") as time_file:
        wall_time, peak_memory = map(float, time_file.read().split())

    runs.append({
        "method": method,
        "timestep": timestep,
        "wall_time": wall_time,
        "peak_memory": peak_memory,
        "errors": errors,
    })

runs.sort(key=lambda run: (run["method"], -run["timestep"]))
print(f"{'method':>8} {'dt (ka)':>8} {'time (s)':>9} {'mem (GiB)':>9} "
      f"{'u (m/a)':>10} {'ρ (kg/m³)':>10} {'h (m)':>10}")
for run in runs:
    e = run["errors"]
    print(f"{run['method']:>8} {run['timestep']:>8g} {run['wall_time']:>9.0f} "
          f"{run['peak_memory'] / 2**20:>9.2f} {e['velocity']:>10.3e} "
          f"{e['density']:>10.3e} {e['thickness']:>10.3e}")

styles = {
    "euler": {"label": "Backward Euler", "color": "tab:blue", "marker": "o"},
    "radau": {"label": "Radau-IIA, 2 stages", "color": "tab:orange", "marker": "s"},
    "rana": {"label": "Radau-IIA, Rana PC", "color": "tab:green", "marker": "^"},
}
orders = {"euler": 1, "radau": 3, "rana": 3}
label_kws = {
    "euler": {"xytext": (4, 4)},
    "radau": {"xytext": (-4, -4), "ha": "right", "va": "top"},
    "rana": {"xytext": (-4, -4), "ha": "right", "va": "top"},
}
panels = {
    "velocity": ("Velocity", "RMS error (m/a)"),
    "density": ("Density", "RMS error (kg/m$^3$)"),
    "thickness": ("Thickness", "RMS error (m)"),
}

print("\nObserved convergence rates between successive timesteps")
print(f"{'method':>8} {'dt (ka)':>12} {'u':>6} {'ρ':>6} {'h':>6}")
for method in styles:
    method_runs = [run for run in runs if run["method"] == method]
    for coarse, fine in zip(method_runs[:-1], method_runs[1:]):
        ratio = coarse["timestep"] / fine["timestep"]
        rates = [
            np.log(coarse["errors"][name] / fine["errors"][name]) / np.log(ratio)
            for name in panels
        ]
        dts = f"{coarse['timestep']:g}→{fine['timestep']:g}"
        print(f"{method:>8} {dts:>12} " + " ".join(f"{rate:>6.2f}" for rate in rates))

fig, axes = plt.subplots(
    ncols=len(panels), sharey=True, figsize=(9.6, 3.2), layout="constrained"
)
axes[0].set_ylabel("Wall time (min)")
for ax, (name, (title, xlabel)) in zip(axes, panels.items()):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.grid(which="major", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(0.1)
    for method, style in styles.items():
        method_runs = [run for run in runs if run["method"] == method]
        errors = [run["errors"][name] for run in method_runs]
        times = [run["wall_time"] / 60 for run in method_runs]
        ax.plot(errors, times, markersize=5, **style)
        for run, error, time in zip(method_runs, errors, times):
            ax.annotate(
                f"{run['timestep']:g}",
                (error, time),
                textcoords="offset points",
                fontsize=7,
                #color="0.35",
                **label_kws[method],
            )

axes[0].legend(
    loc="lower left", fontsize=8, title="Point labels: Δt (ka)", title_fontsize=7
)
axes[0].set_title("(b)", loc="left", fontweight="bold")
fig.savefig(args.output, dpi=300, bbox_inches="tight")

# Convergence plot. The dashed lines show the theoretical order of each method,
# drawn over its two finest timesteps and offset below the data.
fig, axes = plt.subplots(
    ncols=len(panels), sharex=True, figsize=(9.6, 3.2), layout="constrained"
)
for ax, (name, (title, ylabel)) in zip(axes, panels.items()):
    ax.set_title(title)
    ax.set_xlabel("Timestep (ka)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.grid(which="major", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(0.1)
    for method, style in styles.items():
        method_runs = [run for run in runs if run["method"] == method]
        timesteps = np.array([run["timestep"] for run in method_runs])
        errors = np.array([run["errors"][name] for run in method_runs])
        ax.plot(timesteps, errors, markersize=5, **style)

        order = orders[method]
        dts = timesteps[-2:]
        ax.plot(
            dts,
            0.5 * errors[-1] * (dts / dts[-1]) ** order,
            "--",
            color=style["color"],
            linewidth=1,
            alpha=0.6,
        )
        ax.annotate(
            r"$\Delta t$" if order == 1 else rf"$\Delta t^{order}$",
            (np.sqrt(dts[0] * dts[1]), 0.5 * errors[-1] * 2 ** (order / 2)),
            xytext=(2, -12),
            textcoords="offset points",
            fontsize=8,
            #color="0.35",
        )

axes[0].set_title("(a)", loc="left", fontweight="bold")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="outside lower center",
    ncols=len(styles),
    fontsize=8,
)
fig.savefig(args.convergence_output, dpi=300, bbox_inches="tight")
