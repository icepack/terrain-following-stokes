from numpy import pi as π
import tqdm
import pytest
from ufl.algorithms import expand_derivatives
import firedrake
from firedrake import Constant, derivative, dx
import irksome
from zetastokes import terrain_following as tfc


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


@pytest.mark.parametrize(
    "frame",
    [
        True,
        pytest.param(
            False, marks=pytest.mark.xfail(strict=True, reason="need frame")
        )
    ]
)
def test_relaxation(frame: bool):
    h_0 = Constant(constants["mean_thickness"])
    δ = Constant(constants["amplitude"])
    L = Constant(constants["length"])

    g = Constant(constants["gravity"])
    μ = Constant(constants["viscosity"])
    ρ_0 = Constant(constants["density"])

    degree = 2
    u_element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
    p_element = firedrake.FiniteElement("DQ", "quadrilateral", degree - 1)
    cg1 = firedrake.FiniteElement("CG", "interval", 1)
    r = firedrake.FiniteElement("R", "interval", 0)
    h_element = firedrake.TensorProductElement(cg1, r)

    ρdegree = 0  # TODO: a parameter
    ρ_element = firedrake.FiniteElement("DQ", "quadrilateral", ρdegree)

    boundary_data = {"robin_ids": [1, 2], "dirichlet_ids": ["bottom"]}

    method = irksome.BackwardEuler()
    pparams = {"form_compiler_parameters": {"quadrature_degree": 4 * degree}}
    sparams = {
        "solver_parameters": {
            "snes_linesearch_type": "nleqerr",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_7": 5,
            "mat_mumps_icntl_14": 100,
            "snes_stol": 0.0,
        },
    }

    ns = [20, 40, 80]
    for n in ns:
        interval = firedrake.IntervalMesh(n, constants["length"])
        mesh = firedrake.ExtrudedMesh(interval, n // 10, name="domain")

        V = firedrake.VectorFunctionSpace(mesh, u_element)
        Q = firedrake.FunctionSpace(mesh, p_element)
        H = firedrake.FunctionSpace(mesh, h_element)
        R = firedrake.FunctionSpace(mesh, ρ_element)
        Z = V * Q * H * R

        x, ζ = firedrake.SpatialCoordinate(mesh)
        h_expr = h_0 + δ * firedrake.cos(π * x / L)
        h_initial = firedrake.Function(H).interpolate(h_expr)
        ρ_initial = firedrake.Function(R).assign(ρ_0)
        b = firedrake.Function(H)

        z = firedrake.Function(Z)
        z.sub(2).assign(h_initial)
        z.sub(3).assign(ρ_0)

        params = {
            "gravity": g,
            "viscosity": μ,
            "penalty": 2 * degree * (degree + 1),
        }

        u, p, h, ρ = firedrake.split(z)
        fields = {
            "velocity": u,
            "pressure": p,
            "thickness": h,
            "density": ρ,
            "bed": b,
        }

        G = tfc.free_energy_rate(**fields, **params, **boundary_data)
        v, q, φ, η = firedrake.TestFunctions(Z)
        F_momentum = expand_derivatives(derivative(G, u, v) + derivative(G, p, q))
        F_mass = tfc.thickness_equation(**fields)
        F_density = tfc.density_equation(**fields, frame=frame)
        F = F_momentum + F_mass + F_density

        F_initial = (
            F_momentum + (h - h_initial) * φ * dx + (ρ - ρ_initial) * η * dx
        )
        firedrake.solve(F_initial == 0, z, **pparams, **sparams)

        timestep = constants["final_time"] / n
        dt = Constant(timestep)
        t = Constant(0.0)
        solver = irksome.TimeStepper(F, method, t, dt, z, **pparams, **sparams)

        zs = [z.copy(deepcopy=True)]
        for step in tqdm.trange(n):
            solver.advance()
            zs.append(z.copy(deepcopy=True))

        u, p, h, ρ = z.subfunctions
        ρ_min, ρ_max = ρ.dat.data_ro.min(), ρ.dat.data_ro.max()
        assert ρ_max - ρ_min < constants["density"] / n
