import firedrake
from firedrake import Constant
import matplotlib.pyplot as plt
import stokes

nx, nz = 16, 16
initial_mesh = firedrake.UnitSquareMesh(nx, nz, diagonal="crossed")
x, z = firedrake.SpatialCoordinate(initial_mesh)

b_0 = Constant(-1)
b_1 = Constant(-0.5)

s_0 = Constant(1.0)
s_1 = Constant(0.5)

lx = Constant(5.0)

b = (1 - x) * b_0 + x * b_1
s = (1 - x) * s_0 + x * s_1
expr = firedrake.as_vector((lx * x, (1 - z) * b + z * s))
fn_space = initial_mesh.coordinates.function_space()
X = firedrake.Function(fn_space).interpolate(expr)
mesh = firedrake.Mesh(X)

fig, ax = plt.subplots()
ax.set_aspect("equal")
firedrake.triplot(mesh, axes=ax)
ax.legend(loc="upper right")
plt.show()

coefficients = {
    "viscosity": 1.0,
    "gravity": 9.81,
}

cg2 = firedrake.FiniteElement("CG", "triangle", 2)
cg1 = firedrake.FiniteElement("CG", "triangle", 1)

V = firedrake.VectorFunctionSpace(mesh, cg2)
Q = firedrake.FunctionSpace(mesh, cg1)
Z = V * Q

z = firedrake.Function(Z)
u, p = firedrake.split(z)

fields = {"velocity": u, "pressure": p}

dirichlet_ids = [1, 2, 3]
bcs = firedrake.DirichletBC(Z.sub(0), 0, dirichlet_ids)

G = stokes.free_energy_rate_cartesian(**fields, **coefficients)
F = firedrake.derivative(G, z)
firedrake.solve(F == 0, z, bcs=bcs)

fig, ax = plt.subplots()
ax.set_aspect("equal")
arrows = firedrake.quiver(z.sub(0), axes=ax)
fig.colorbar(arrows)
plt.show()
