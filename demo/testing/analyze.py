import argparse
import numpy as np
import matplotlib.pyplot as plt
import ufl
import firedrake
from firedrake import norm, Constant, dot
from zetastokes.terrain_following import coordinate_transformation_derivatives
import topography


def terrain_following_to_cartesian(mesh_xyz, u, p, b, h):
    J, J_inv = coordinate_transformation_derivatives(b, h)
    # TODO: Check if this should really be `project` or possibly also
    # interpolate into a DG space.
    Ju = firedrake.Function(u.function_space()).interpolate(dot(J, u))

    V = firedrake.FunctionSpace(mesh_xyz, u.ufl_element())
    Q = firedrake.FunctionSpace(mesh_xyz, p.ufl_element())

    v = firedrake.Function(V)
    v.dat.data[:] = Ju.dat.data_ro[:]
    q = firedrake.Function(Q)
    q.dat.data[:] = p.dat.data_ro[:]

    return v, q


def load_meshes(checkpoint, coords: str):
    nxs = checkpoint.h5pyfile.attrs["nxs"]
    prefix = "domain_" if coords == "xyz" else ""
    return nxs, [checkpoint.load_mesh(name=f"{prefix}rect_{nx}") for nx in nxs]


def load_data(checkpoint, num_cells, meshes, config: str):
    us, ps = [], []
    for nx, mesh in zip(num_cells, meshes):
        us.append(checkpoint.load_function(mesh, name=f"u_{config}_{nx}"))
        ps.append(checkpoint.load_function(mesh, name=f"p_{config}_{nx}"))

    return us, ps


def plot_errors(ax, mesh_spacings, velocity_errors, pressure_errors):
    uslope, uintercept = np.polyfit(np.log(mesh_spacings), np.log(velocity_errors), 1)
    pslope, pintercept = np.polyfit(np.log(mesh_spacings), np.log(pressure_errors), 1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(mesh_spacings, velocity_errors, color="tab:blue", label="Velocity errors")
    ax.scatter(mesh_spacings, pressure_errors, color="tab:orange", label="Pressure errors")

    ucurve = np.exp(uintercept) * mesh_spacings ** uslope
    pcurve = np.exp(pintercept) * mesh_spacings ** pslope
    ulabel = f"log|U₁ - U₂| ~ {uintercept:0.2f} + {uslope:0.2f}·δx"
    plabel = f"log|P₁ - P₂| ~ {pintercept:0.2f} + {pslope:0.2f}·δx"
    ax.plot(mesh_spacings, ucurve, "--", color="tab:blue", label=ulabel)
    ax.plot(mesh_spacings, pcurve, "--", color="tab:orange", label=plabel)

    ax.legend(loc="lower right")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Mesh spacing")


def relative_error(p, q):
    return norm(p - q) / norm(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--output")
    args = parser.parse_args()

    # Load in all the data
    solutions = {}
    with firedrake.CheckpointFile(args.filename, "r") as chk:
        topo_name = chk.h5pyfile.attrs["topography"]
        nxs, meshes_xyz = load_meshes(chk, "xyz")
        solutions["xyz"] = {
            "cg": load_data(chk, nxs, meshes_xyz, "xyz_cg"),
            "hdiv": load_data(chk, nxs, meshes_xyz, "xyz_hdiv"),
        }

    with firedrake.CheckpointFile(args.filename, "r") as chk:
        nxs, meshes_tfc = load_meshes(chk, "tfc")
        solutions["tfc"] = {
            "cg": {
                "inf": load_data(chk, nxs, meshes_tfc, "tfc_inf_cg"),
                1: load_data(chk, nxs, meshes_tfc, "tfc_1_cg"),
                2: load_data(chk, nxs, meshes_tfc, "tfc_2_cg"),
            },
            "hdiv": {
                "inf": load_data(chk, nxs, meshes_tfc, "tfc_inf_hdiv"),
                1: load_data(chk, nxs, meshes_tfc, "tfc_1_hdiv"),
                2: load_data(chk, nxs, meshes_tfc, "tfc_2_hdiv"),
            }
        }

    # Compare CG vs DG in Cartesian coordinates
    us_cg, ps_cg = solutions["xyz"]["cg"]
    us_hdiv, ps_hdiv = solutions["xyz"]["hdiv"]
    u_errors = [relative_error(*us) for us in zip(us_cg, us_hdiv)]
    p_errors = [relative_error(*ps) for ps in zip(ps_cg, ps_hdiv)]

    fig, ax = plt.subplots()
    plot_errors(ax, 1 / nxs, u_errors, p_errors)
    ax.set_title("XYZ: (CG₂)² x CG₁  vs  BDMCF x DG₀")
    fig.savefig(f"{args.output}-01-xyz_cg2-cg1-vs-bdmcf1-dg0.png", bbox_inches="tight")

    # Compare CG in Cartesian vs terrain-following coordinates
    us_tfc, ps_tfc = solutions["tfc"]["cg"]["inf"]

    us_tfc2xyz, ps_tfc2xyz = [], []
    for mesh_xyz, mesh_tfc, u, p in zip(meshes_xyz, meshes_tfc, us_tfc, ps_tfc):
        x, ζ = firedrake.SpatialCoordinate(mesh_tfc)
        lx = Constant(mesh_tfc.coordinates.dat.data_ro[:, 0].max())
        topo_fn = getattr(topography, topo_name)
        b, h = topo_fn(x / lx)

        v, q = terrain_following_to_cartesian(mesh_xyz, u, p, b, h)
        us_tfc2xyz.append(v)
        ps_tfc2xyz.append(q)

    u_errors = [relative_error(*us) for us in zip(us_cg, us_tfc2xyz)]
    p_errors = [relative_error(*ps) for ps in zip(ps_cg, ps_tfc2xyz)]

    fig, ax = plt.subplots()
    plot_errors(ax, 1 / nxs, u_errors, p_errors)
    ax.set_title("(CG₂)² x CG₁ in XYZ vs TFC")
    fig.savefig(f"{args.output}-02-cg2-cg1_xyz-vs-tfc.png", bbox_inches="tight")

    # Compare CG vs DG in terrain-following coordinates
    us_tfc_dg, ps_tfc_dg = solutions["tfc"]["hdiv"]["inf"]

    u_errors = [relative_error(*us) for us in zip(us_tfc, us_tfc_dg)]
    p_errors = [relative_error(*ps) for ps in zip(ps_tfc, ps_tfc_dg)]

    fig, ax = plt.subplots()
    plot_errors(ax, 1 / nxs, u_errors, p_errors)
    ax.set_title("TFC: (CG₂)² x CG₁  vs  BDMCF x DG₀")
    fig.savefig(f"{args.output}-03-tfc_cg2-cg1-vs-bdmcf1-dg0.png", bbox_inches="tight")

    # Compare CG with exact and discretized terrain-following coordinates
    us_tfc_hcg, ps_tfc_hcg = solutions["tfc"]["cg"][2]

    u_errors = [relative_error(*us) for us in zip(us_tfc, us_tfc_hcg)]
    p_errors = [relative_error(*ps) for ps in zip(ps_tfc, ps_tfc_hcg)]

    fig, ax = plt.subplots()
    plot_errors(ax, 1 / nxs, u_errors, p_errors)
    ax.set_title("TFC / (CG₂)² x CG₁: exact vs discrete coords")
    fig.savefig(f"{args.output}-04-tfc-cg2-cg1_exacth-vs-discreteh.png", bbox_inches="tight")

    # Compare DG with exact and discretized terrain-following coordinates
    us_tfc_dg_hcg, ps_tfc_dg_hcg = solutions["tfc"]["hdiv"][2]

    u_errors = [relative_error(*us) for us in zip(us_tfc_dg, us_tfc_dg_hcg)]
    p_errors = [relative_error(*ps) for ps in zip(ps_tfc_dg, ps_tfc_dg_hcg)]

    fig, ax = plt.subplots()
    plot_errors(ax, 1 / nxs, u_errors, p_errors)
    ax.set_title("TFC / BDMCF x DG₀: exact vs discrete coords")
    fig.savefig(f"{args.output}-05-tfc-bdmcg1-dg0_exacth-vs-discreteh.png", bbox_inches="tight")
