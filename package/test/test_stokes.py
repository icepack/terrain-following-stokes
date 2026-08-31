from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatchmethod, partial
import pytest
import numpy as np
from numpy import pi as π
import firedrake
from firedrake import Constant, dot, Function, FiniteElement, MeshGeometry
import ufl
from zetastokes import cartesian, terrain_following


lx = 5.0
constants = {"viscosity": 1.0, "density": 1.0, "gravity": 9.81}
fcparams = {"quadrature_degree": 8}
sparams = {"pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def topography(x):
    b_0 = Constant(0.0)
    δb = Constant(1 / 8)
    k_b = Constant(3)
    δx_b = Constant(1 / 7)
    b = b_0 + δb * firedrake.cos(2 * π * k_b * (x - δx_b))

    s_0 = Constant(1.0)
    δs = Constant(1 / 8)
    k_s = Constant(5)
    δx_s = Constant(3 / 2)
    s = s_0 + δs * firedrake.cos(2 * π * k_s * (x - δx_s))

    return b, s - b


@dataclass
class CartesianCoordinates:
    topography: Callable = topography
    length: float = lx
    aspect_ratio: float = 5

    @singledispatchmethod
    def mesh(self, arg):
        pass

    @mesh.register
    def _(self, arg: MeshGeometry) -> MeshGeometry:
        x, ζ = firedrake.SpatialCoordinate(arg)
        b, h = self.topography(x / Constant(self.length))
        expr = firedrake.as_vector((x, b + h * ζ))
        Vc = arg.coordinates.function_space()
        X = Function(Vc).interpolate(expr)
        return firedrake.Mesh(X)

    @mesh.register
    def _(self, arg: np.int64 | int):
        lz = (lx := self.length) / self.aspect_ratio
        nz = int(arg / self.aspect_ratio)
        interval = firedrake.IntervalMesh(arg, lx)
        initial_mesh = firedrake.ExtrudedMesh(interval, nz)
        return self.mesh(initial_mesh)

    def fields(self, mesh):
        return {}

    @property
    def module(self):
        return cartesian


@dataclass
class TerrainFollowingCoordinates:
    topography: Callable = topography
    length: float = lx
    aspect_ratio: float = 5
    degree: int = "inf"

    def mesh(self, nx: np.int64 | int) -> MeshGeometry:
        lz = (lx := self.length) / self.aspect_ratio
        nz = int(nx / self.aspect_ratio)
        interval = firedrake.IntervalMesh(nx, lx)
        return firedrake.ExtrudedMesh(interval, nz)

    def fields(self, mesh):
        x, ζ = firedrake.SpatialCoordinate(mesh)
        b_expr, h_expr = self.topography(x / Constant(self.length))
        if self.degree == "inf":
            return {"thickness": h_expr, "bed": b_expr}

        cg = FiniteElement("CG", "interval", self.degree)
        r = FiniteElement("R", "interval", 0)
        topo_element = firedrake.TensorProductElement(cg, r)
        Q = firedrake.FunctionSpace(mesh, topo_element)
        h = Function(Q).interpolate(h_expr)
        b = Function(Q).interpolate(b_expr)
        return {"thickness": h, "bed": b}

    @property
    def module(self):
        return terrain_following


class CGBasis:
    def function_spaces(self, mesh):
        u_elt = FiniteElement("CG", "quadrilateral", 2)
        p_elt = FiniteElement("CG", "quadrilateral", 1)
        V = firedrake.VectorFunctionSpace(mesh, u_elt, name="velocity")
        Q = firedrake.FunctionSpace(mesh, p_elt, name="pressure")
        return V, Q

    def free_energy_rate(self, module):
        return module.cell_free_energy_rate


@dataclass
class PrimalDGBasis:
    degree: int

    def function_spaces(self, mesh: firedrake.MeshGeometry):
        u_elt = FiniteElement("DQ", "quadrilateral", self.degree + 1)
        p_elt = FiniteElement("DQ", "quadrilateral", self.degree)
        V = firedrake.VectorFunctionSpace(mesh, u_elt, name="velocity")
        Q = firedrake.FunctionSpace(mesh, p_elt, name="pressure")
        return V, Q

    def free_energy_rate(self, module):
        return module.free_energy_rate


@dataclass
class DualDGBasis:
    degree: int

    def function_spaces(self, mesh: firedrake.MeshGeometry):
        u_elt = FiniteElement("DQ", "quadrilateral", self.degree + 1)
        p_elt = FiniteElement("DQ", "quadrilateral", self.degree)
        S = firedrake.TensorFunctionSpace(mesh, p_elt, symmetry=True, name="stress")
        V = firedrake.VectorFunctionSpace(mesh, u_elt, name="velocity")
        Q = firedrake.FunctionSpace(mesh, p_elt, name="pressure")
        return V, Q, S

    def free_energy_rate(self, module):
        return partial(module.free_energy_rate, form="dual")


@dataclass
class StokesSolver:
    basis: CGBasis | PrimalDGBasis | DualDGBasis
    coords: CartesianCoordinates | TerrainFollowingCoordinates

    def solve(self, mesh):
        fn_spaces = self.basis.function_spaces(mesh)
        names = [space.name for space in fn_spaces]
        Z = firedrake.MixedFunctionSpace(fn_spaces)
        z = Function(Z)

        fields = (
            {name: field for name, field in zip(names, firedrake.split(z))} |
            self.coords.fields(mesh)
        )

        # FIXME: make option for Robin BCs
        dirichlet_ids = [1, 2, "bottom"]
        # FIXME: estimate based on degree
        params = {"form_compiler_parameters": fcparams, "solver_parameters": sparams}
        if isinstance(self.basis, CGBasis):
            velocity_index = names.index("velocity")
            bcs = firedrake.DirichletBC(Z.sub(velocity_index), 0, dirichlet_ids)
            params["bcs"] = bcs

        boundary_data = {"dirichlet_ids": dirichlet_ids}
        fn = self.basis.free_energy_rate(self.coords.module)
        G = fn(**fields, **constants, **boundary_data)

        F = firedrake.derivative(G, z)
        firedrake.solve(F == 0, z, **params)
        return z


def tfc_to_xyz(
    u: Function, coords: TerrainFollowingCoordinates, mesh_xyz: MeshGeometry
) -> Function:
    # NOTE that we might need to be careful about the degree we're
    # projecting into here. If the thickness is discretized in CG(k) then
    # we need to add some extra degree in how we represent the Cartesian
    # velocity because `u` gets multiplied by the coordinate transformation
    # `J` which includes a factor of `h`. If the thickness is not
    # discretized and we use an analytical expression which might contain
    # e.g. transcendental functions then we arguably need even more room.
    degree = max(*u.ufl_element().degree())
    hdegree = 2  # FIXME
    u_elt = FiniteElement("DQ", "quadrilateral", degree + hdegree)

    # Create the Cartesian velocity, defined on the terrain-following mesh
    mesh_tfc = ufl.domain.extract_unique_domain(u)
    x, ζ = firedrake.SpatialCoordinate(mesh_tfc)
    b, h = coords.topography(x / Constant(coords.length))
    J, J_inv = terrain_following.coordinate_transformation_derivatives(b, h)
    V_dg = firedrake.VectorFunctionSpace(mesh_tfc, u_elt)
    Ju = Function(V_dg).project(dot(J, u))

    # Transfer the Cartesian velocity to the Cartesian mesh
    V = firedrake.VectorFunctionSpace(mesh_xyz, u_elt)
    u_xyz = Function(V)
    u_xyz.dat.data[:] = Ju.dat.data_ro[:]

    return u_xyz


def relative_error(q1, q2):
    return firedrake.norm(q1 - q2) / firedrake.norm(q2)


def run_suite(config1: dict, config2: dict):
    nxs = np.array([16, 20, 24, 32, 48, 64, 72, 84, 96, 108, 128])
    meshes1 = [config1["coords"].mesh(nx) for nx in nxs]
    # If the two configurations have the same coordinate system, do nothing.
    # Otherwise, generate the 2nd configurations meshes based on the first,
    # and create a function that maps velocities from the one coordinate system
    # to the other.
    if isinstance(config2["coords"], type(config1["coords"])):
        meshes2 = meshes1
        transfer_fn = lambda u, mesh: u
    else:
        meshes2 = [config2["coords"].mesh(mesh) for mesh in meshes1]
        transfer_fn = lambda u, mesh: tfc_to_xyz(u, config1["coords"], mesh)

    solver1 = StokesSolver(**config1)
    solver2 = StokesSolver(**config2)

    zs1 = [solver1.solve(mesh) for mesh in meshes1]
    zs2 = [solver2.solve(mesh) for mesh in meshes2]

    us1_ = [z.subfunctions[0] for z in zs1]
    us2 = [z.subfunctions[0] for z in zs2]
    us1 = [transfer_fn(u, mesh) for u, mesh in zip(us1_, meshes2)]

    errors = np.array([relative_error(u1, u2) for u1, u2 in zip(us1, us2)])
    slope, intercept = np.polyfit(np.log(1 / nxs), np.log(errors), 1)
    print(f"log |u₁ - u₂| / |u₂| ~ {intercept:0.2f} + {slope:0.2f} * log(δx)")
    assert slope > 0.9


xyz = CartesianCoordinates()
tfcx = TerrainFollowingCoordinates(degree="inf")
tfc1 = TerrainFollowingCoordinates(degree=1)
tfc2 = TerrainFollowingCoordinates(degree=2)

cg = CGBasis()
pdg1 = PrimalDGBasis(degree=1)
ddg1 = DualDGBasis(degree=1)

configurations = {
    "xyz_cg": {"coords": xyz, "basis": cg},
    "xyz_dg": {"coords": xyz, "basis": pdg1},
    "xyz_ddg": {"coords": xyz, "basis": ddg1},
    "tfc_ddg1": {"coords": tfcx, "basis": ddg1},
    "tfc_cg": {"coords": tfcx, "basis": cg},
    "tfc_h1": {"coords": tfc1, "basis": pdg1},
    "tfc_h2": {"coords": tfc2, "basis": pdg1},
}


@pytest.mark.parametrize(
    "config1, config2",
    [
        ("xyz_cg", "xyz_dg"),
        ("tfc_cg", "xyz_cg"),
        ("tfc_h1", "xyz_cg"),
        ("tfc_h2", "xyz_cg"),
        ("xyz_ddg", "xyz_dg"),
        ("tfc_ddg1", "tfc_h1"),
    ],
)
def test_stokes(config1, config2):
    run_suite(configurations[config1], configurations[config2])
