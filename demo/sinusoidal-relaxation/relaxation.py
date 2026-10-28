r"""Relaxation of a sinusoidal surface; experiment from Ahlkrona et al. 2026"""

import argparse
import sys
import numpy as np
from numpy import pi as π
import tqdm
import firedrake
from firedrake import Constant, derivative, dx
from ufl.algorithms import expand_derivatives
import irksome
from zetastokes import cartesian, terrain_following


parser = argparse.ArgumentParser()
parser.add_argument("--lag-u", action="store_true")
parser.add_argument("--lag-h", action="store_true")
parser.add_argument("--method", choices=["euler", "radau"], default="euler")
parser.add_argument("--final-time", type=float, default=20.0)
parser.add_argument("--num-steps", type=int, default=40)
parser.add_argument("--output", default="relaxation")
args = parser.parse_args()


sec_per_year = 365.25 * 24 * 60 * 60

constants = {
    "length": 100e3,                              # meters
    "mean_thickness": 1e3,
    "amplitude": 1e2,
    "viscosity": 1e12 / (1e6 * sec_per_year),     # megapascals * years
    "density": 910,                               # kg / meter^3
    "gravity": 9.81 / 1e6,                        # so ρg is in MPa / meter
    "final_time": 20,                             # years
}


sparams = {
    "solver_parameters": {
        "snes_linesearch_type": "nleqerr",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_7": 5,
        "mat_mumps_icntl_14": 100,
        "snes_stol": 0.0,
        "snes_monitor": f"ascii:{args.output}.log",
    },
}


def initial_surface(x, *, mean_thickness, amplitude, length, **kwargs):
    H = Constant(mean_thickness)
    δ = Constant(amplitude)
    L = Constant(length)
    return H + δ * firedrake.cos(π * x / L)


# Set up the mesh and function spaces
nx, nz = 50, 5
interval = firedrake.IntervalMesh(nx, constants["length"])
mesh = firedrake.ExtrudedMesh(interval, nz, name="domain")

degree = 2
u_element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
p_element = firedrake.FiniteElement("DQ", "quadrilateral", degree - 1)
V = firedrake.VectorFunctionSpace(mesh, u_element)
Q = firedrake.FunctionSpace(mesh, p_element)

cg1 = firedrake.FiniteElement("CG", "interval", 1)
r = firedrake.FiniteElement("R", "interval", 0)
h_element = firedrake.TensorProductElement(cg1, r)
H = firedrake.FunctionSpace(mesh, h_element)

Z = V * Q * H

# Create the initial data
x, ζ = firedrake.SpatialCoordinate(mesh)
h_expr = initial_surface(x, **constants)
h_initial = firedrake.Function(H).interpolate(h_expr)
b = firedrake.Function(H)

z = firedrake.Function(Z)
z.sub(2).assign(h_initial)
u, p, h = firedrake.split(z)

# Set up the momentum and mass balance equations
params = {
    "density": constants["density"],
    "gravity": constants["gravity"],
    "penalty": 2 * degree * (degree + 1),
    "viscosity": constants["viscosity"],
}
fields = {
    "velocity": u,
    "pressure": p,
    "thickness": irksome.lag(h) if args.lag_h else h,
    "bed": b,
}
# TODO: Find out what boundary conditions Alkhrona et al. used. At the least
# we should have free slip along the side walls but the bed could be either no
# slip or friction.
boundary_data = {"robin_ids": [1, 2], "dirichlet_ids": ["bottom"]}
G = terrain_following.free_energy_rate(**fields, **params, **boundary_data)

v, q, φ = firedrake.TestFunctions(Z)
F_momentum = expand_derivatives(derivative(G, u, v) + derivative(G, p, q))

fields = {"velocity": irksome.lag(u) if args.lag_u else u, "thickness": h}
F_mass = terrain_following.thickness_equation(**fields)

F = F_momentum + F_mass

# Do an initial solve of the momentum balance equation by itself in order to
# get the correct starting velocity and pressure
pparams = {"form_compiler_parameters": {"quadrature_degree": 4 * degree}}
F_initial = F_momentum + (h - h_initial) * φ * dx
firedrake.solve(F_initial == 0, z, **pparams, **sparams)

u = z.subfunctions[0]
u_max = np.abs(u.dat.data_ro).max()
δx = constants["length"] / nx
cfl_time = δx / u_max
print(f"CFL time: {cfl_time} years")

# Step the model forward in time
methods = {"euler": irksome.BackwardEuler(), "radau": irksome.RadauIIA(2)}
method = methods[args.method]
t = Constant(0.0)
timestep = args.final_time / args.num_steps
dt = Constant(timestep)
solver = irksome.TimeStepper(F, method, t, dt, z, **pparams, **sparams)

zs = [z.copy(deepcopy=True)]
with firedrake.CheckpointFile(f"{args.output}.h5", "w") as chk:
    chk.save_mesh(mesh)
    chk.h5pyfile.attrs["num_steps"] = args.num_steps + 1
    chk.h5pyfile.attrs["final_time"] = args.final_time
    chk.save_function(z, name="solution", idx=0)
    for step in tqdm.trange(args.num_steps):
        try:
            solver.advance()
            zs.append(z.copy(deepcopy=True))
            chk.save_function(z, name="solution", idx=step + 1)
        except firedrake.ConvergenceError as error:
            print(f"Failed at step {step}!")
            chk.h5pyfile.attrs["num_steps"] = step
            chk.h5pyfile.attrs["final_time"] = step * timestep
            chk.h5pyfile.attrs["success"] = False
            sys.exit(0)

    chk.h5pyfile.attrs["success"] = True
