import numpy as np
from numpy import pi as π
import tqdm
import firedrake
from firedrake import as_vector, Constant, dot, derivative, dx
from ufl.algorithms import expand_derivatives
import irksome
from zetastokes.terrain_following import (
    free_energy_rate, thickness_equation, density_equation,
    coordinate_transformation_derivatives
)


sec_per_year = 365.25 * 24 * 60 * 60

constants = {
    "length": 500,                                   # kilometers
    "thickness": 500,
    "depth": 100,
    "amplitude": 10,
    "viscosity": 1e20 / (1e9 * 1e3 * sec_per_year),  # gigapascals * millennia
    "density": 3.2e3,                                # kg / meter^3
    "delta_density": 100,
    "gravity": 9.81 / 1e9 * 1e3,                     # so ρg is in GPa / km
    "final_time": 800,                               # millennia
}


sparams = {
    "solver_parameters": {
        "snes_linesearch_type": "nleqerr",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_7": 5,
        "mat_mumps_icntl_14": 100,
        "snes_stol": 0.0,
        "snes_monitor": None,
        "snes_linesearch_monitor": None,
    },
}


def initial_density(x, z):
    z_0 = Constant(constants["thickness"] - constants["depth"])
    δz = Constant(constants["amplitude"])
    L = Constant(constants["length"])
    Z = z_0 + δz * firedrake.sin(π * x / L)
    ρ_0 = Constant(constants["density"])
    δρ = Constant(constants["delta_density"])
    return firedrake.conditional(z <= Z, ρ_0, ρ_0 + δρ)


nx = 50
nz = int(constants["thickness"] / constants["length"] * nx)
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

ρ_element = firedrake.FiniteElement("DQ", "quadrilateral", 0)
R = firedrake.FunctionSpace(mesh, ρ_element)
x, ζ = firedrake.SpatialCoordinate(mesh)
z = Constant(constants["thickness"]) * ζ
ρ_initial = firedrake.Function(R).project(initial_density(x, z))

h_initial = firedrake.Function(H).assign(constants["thickness"])
b = firedrake.Function(H)

Z = V * Q * R * H
z = firedrake.Function(Z)

z.sub(2).assign(ρ_initial)
z.sub(3).assign(h_initial)

u, p, ρ, h = firedrake.split(z)

params = {
    "gravity": constants["gravity"],
    "penalty": 2 * degree * (degree + 1),
    "viscosity": constants["viscosity"],
}
fields = {"velocity": u, "pressure": p, "bed": b, "thickness": h, "density": ρ}
boundary_data = {"robin_ids": [1, 2], "dirichlet_ids": ["bottom"]}
G = free_energy_rate(**fields, **params, **boundary_data)

v, q, η, φ = firedrake.TestFunctions(Z)
F_momentum = expand_derivatives(derivative(G, u, v) + derivative(G, p, q))
F_mass = thickness_equation(**fields)
F_density = density_equation(**fields)
F = F_momentum + F_mass + F_density

pparams = {"form_compiler_parameters": {"quadrature_degree": 4 * degree}}
F_initial = F_momentum + (ρ - ρ_initial) * η * dx + (h - h_initial) * φ * dx
firedrake.solve(F_initial == 0, z, **pparams, **sparams)

u = z.subfunctions[0]
J, _ = coordinate_transformation_derivatives(b, h_initial)
u_xyz = firedrake.Function(V).interpolate(dot(J, u))
u_max = np.abs(u_xyz.dat.data_ro).max()
δx = constants["length"] / nx
cfl_time = δx / u_max
print(f"CFL time: {cfl_time} ka")

# Step the model forward in time
method = irksome.BackwardEuler()
t = Constant(0.0)
timestep = 5.0
dt = Constant(timestep)
solver = irksome.TimeStepper(F, method, t, dt, z, **pparams, **sparams)

zs = [z.copy(deepcopy=True)]
num_steps = int(constants["final_time"] / timestep)
with firedrake.CheckpointFile("rayleigh-taylor.h5", "w") as chk:
    chk.save_mesh(mesh)
    chk.h5pyfile.attrs["final_time"] = constants["final_time"]
    chk.save_function(z, name="solution", idx=0)
    try:
        for step in tqdm.trange(num_steps):
            solver.advance()
            zs.append(z.copy(deepcopy=True))
            chk.save_function(z, name="solution", idx=step + 1)
            chk.h5pyfile.attrs["num_steps"] = step + 1
    except firedrake.ConvergenceError as error:
        print(f"Failed at step {step}!")
        raise error
