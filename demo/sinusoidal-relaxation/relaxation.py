r"""Relaxation of a sinusoidal surface; experiment from Ahlkrona et al. 2026"""

import numpy as np
from numpy import pi as π
import tqdm
import firedrake
from firedrake import Constant, derivative, dx
from ufl.algorithms import expand_derivatives
import irksome
from zetastokes import cartesian, terrain_following


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
        "snes_monitor": "ascii:relaxation.log",
    },
}


def initial_surface(x, *, mean_thickness, amplitude, length, **kwargs):
    H = Constant(mean_thickness)
    δ = Constant(amplitude)
    L = Constant(length)
    return H + δ * firedrake.cos(π * x / L)


timesteps = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 20])

# Set up the mesh and function spaces
nx, nz = 50, 5
interval = firedrake.IntervalMesh(nx, constants["length"])
mesh = firedrake.ExtrudedMesh(interval, nz)

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
    "gravity": constants["density"] * constants["gravity"],
    "penalty": 2 * degree * (degree + 1),
    "viscosity": constants["viscosity"],
}
fields = {"velocity": u, "pressure": p, "thickness": h, "bed": b}
# TODO: Find out what boundary conditions Alkhrona et al. used. At the least
# we should have free slip along the side walls but the bed could be either no
# slip or free slip.
boundary_data = {"robin_ids": [1, 2], "dirichlet_ids": ["bottom"]}
G = terrain_following.free_energy_rate(**fields, **params, **boundary_data)

v, q, φ = firedrake.TestFunctions(Z)
F_momentum = expand_derivatives(derivative(G, u, v) + derivative(G, p, q))
F_mass = terrain_following.mass_balance(**fields)
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
method = irksome.RadauIIA(2)
t = Constant(0.0)
dt = Constant(0.5)
solver = irksome.TimeStepper(F, method, t, dt, z, **pparams, **sparams)

zs = [z.copy(deepcopy=True)]
num_steps = int(constants["final_time"] / float(dt))
for step in tqdm.trange(num_steps):
    solver.advance()
    zs.append(z.copy(deepcopy=True))
