r"""Spin-up of a glacier with random bed topography; experiment from Ahlkrona
et al. 2026"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import tqdm
import firedrake
from firedrake import Constant, derivative, inner, dx, ds_b
from ufl.algorithms import expand_derivatives
import irksome
from zetastokes.terrain_following import free_energy_rate, thickness_equation
import noise


sec_per_year = 365.25 * 24 * 60 * 60

constants = {
    "length": 8e3,
    "slope": 0.1,
    "viscosity": 1e12 / (1e6 * sec_per_year),     # megapascals * years
    "density": 910,                               # kg / meter^3
    "gravity": 9.81 / 1e6,                        # so ρg is in MPa / meter
    "friction": 0.1,                              # megapascals * years / meters
}


sparams = {
    "solver_parameters": {
        "snes_type": "vinewtonrsls",
        "snes_linesearch_type": "secant",
        "snes_linesearch_max_it": 40,
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_7": 5,
        "mat_mumps_icntl_14": 100,
        "snes_stol": 0.0,
        "snes_atol": 1e-10,
        "snes_monitor": None,
        "snes_converged_reason": None,
    },
}


# Set up the mesh and function spaces
nx, nz = 500, 5
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

# Create some random bed topography
b = firedrake.Function(H)
xs = interval.coordinates.dat.data_ro
L, α = constants["length"], constants["slope"]
rng = np.random.default_rng(seed=1729)
cs = np.array([0.25, 0.5, 1.0, 1.0])
generator = noise.PerlinNoise(cs, rng)
bs = α / L * (xs - L)**2 + α * L * generator(xs / L)
b.dat.data[:] = bs

h_min = Constant(10.0)
h_initial = firedrake.Function(H).assign(h_min)
z = firedrake.Function(Z)
z.sub(2).assign(h_initial)

# Initial momentum solve
u, p, h = firedrake.split(z)
params = {
    "density": constants["density"],
    "gravity": constants["gravity"],
    "penalty": 2 * degree * (degree + 1),
    "viscosity": constants["viscosity"],
}
fields = {"velocity": u, "pressure": p, "thickness": h, "bed": b}
boundary_data = {"dirichlet_ids": [1, 2], "robin_ids": ["bottom"]}
C = Constant(constants["friction"])
n = firedrake.FacetNormal(mesh)
G = (
    free_energy_rate(**fields, **params, **boundary_data) +
    0.5 * C * inner(u, u) * ds_b
)

v, q, φ = firedrake.TestFunctions(Z)
F_momentum = expand_derivatives(derivative(G, u, v) + derivative(G, p, q))

pparams = {"form_compiler_parameters": {"quadrature_degree": 4 * degree}}
F_initial = F_momentum + (h - h_initial) * φ * dx
firedrake.solve(F_initial == 0, z, **pparams, **sparams)

# Create the mass balance function
x, ζ = firedrake.SpatialCoordinate(mesh)
s = b + h

α = Constant(constants["slope"])
L = Constant(constants["length"])
s_max = Constant(α * L)
a_max = Constant(1.0)
ela_fraction = Constant(0.4)  # Equilibrium is at this fraction of max height
da_ds = Constant(a_max / ((1 - ela_fraction) * s_max))
a_expr = firedrake.min_value(a_max, da_ds * (s - ela_fraction * s_max))
a = firedrake.Function(H).interpolate(a_expr)

# Time evolution
u = z.subfunctions[0]
u_max = np.abs(u.dat.data_ro).max()
δx = constants["length"] / nx
cfl_time = δx / u_max
print(f"CFL time: {cfl_time} years")

F_mass = thickness_equation(**fields, outflow_ids=[1, 2]) - a * φ * dx
F = F_momentum + F_mass

lower, upper = firedrake.Function(Z), firedrake.Function(Z)
lower.assign(-np.inf), upper.assign(+np.inf)
lower.sub(2).assign(h_min)
bounds = ("stage", lower, upper)
bparams = {"stage_type": "value", "basis_type": "Bernstein", "bounds": bounds}

method = irksome.BackwardEuler()
t = Constant(0.0)
timestep = 1.0
final_time = 400.0
num_steps = int(final_time / timestep)
dt = Constant(timestep)
solver = irksome.TimeStepper(F, method, t, dt, z, **pparams, **sparams, **bparams)

with firedrake.CheckpointFile("perlin.h5", "w") as chk:
    chk.h5pyfile.attrs["num_steps"] = num_steps + 1
    chk.h5pyfile.attrs["final_time"] = final_time
    chk.save_mesh(mesh)
    chk.save_function(b, name="bed")
    chk.save_function(z, name="solution", idx=0)
    for step in tqdm.trange(num_steps):
        try:
            solver.advance()
            a.interpolate(a_expr)
            chk.save_function(z, name="solution", idx=step + 1)
        except firedrake.ConvergenceError as error:
            print(f"Crashed at step {step}!")
            chk.h5pyfile.attrs["num_steps"] = step
            chk.h5pyfile.attrs["final_time"] = step * timestep
            raise error
