import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_FILE = PROJECT_ROOT / "data" / "itokawa_50k_ascii.ply"
OUTPUT_DIR = PROJECT_ROOT / "results"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.integrate import solve_ivp

from eom import (
    OMEGA_VEC,
    ROTATION_PERIOD_S,
    hamiltonian_to_lagrangian,
    lagrangian_eom,
    lagrangian_to_hamiltonian,
    load_gravity_model,
)


# Initial state in the Itokawa-fixed rotating frame.
# Position is in meters and velocity is in meters per second.
INITIAL_POSITION_M = np.array([1000.0, 0.0, 0.0])
INITIAL_VELOCITY_M_S = np.array([0.0, -0.1, 0.0])

TIME_SPAN_S = (0.0, 2.0 * ROTATION_PERIOD_S)
NUM_OUTPUT_POINTS = 1000
STM_TIME_SPAN_S = (0.0, 0.05 * ROTATION_PERIOD_S)
STM_OUTPUT_POINTS = 100


def propagate_orbit(gravity_model, initial_position, initial_velocity):
    state0 = np.hstack((initial_position, initial_velocity))
    output_times = np.linspace(TIME_SPAN_S[0], TIME_SPAN_S[1], NUM_OUTPUT_POINTS)

    return solve_ivp(
        fun=lambda t, state: lagrangian_eom(t, state, gravity_model),
        t_span=TIME_SPAN_S,
        y0=state0,
        method="DOP853",
        t_eval=output_times,
        rtol=1e-10,
        atol=1e-12,
    )


def propagate_orbit_for_stm(gravity_model, initial_state):
    output_times = np.linspace(STM_TIME_SPAN_S[0], STM_TIME_SPAN_S[1], STM_OUTPUT_POINTS)

    return solve_ivp(
        fun=lambda t, state: lagrangian_eom(t, state, gravity_model),
        t_span=STM_TIME_SPAN_S,
        y0=initial_state,
        method="DOP853",
        t_eval=output_times,
        rtol=1e-11,
        atol=1e-13,
    )


def plot_orbit(solution):
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh = pv.read(MESH_FILE)
    asteroid_points_m = np.asarray(mesh.points) * 1000.0

    q = solution.y[0:3, :]
    radius = np.linalg.norm(q, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        asteroid_points_m[::30, 0],
        asteroid_points_m[::30, 1],
        s=1,
        color="0.7",
        label="Itokawa shape projection",
    )
    axes[0].plot(q[0, :], q[1, :], linewidth=1.5, label="Spacecraft trajectory")
    axes[0].scatter(q[0, 0], q[1, 0], color="green", s=40, label="Start")
    axes[0].scatter(q[0, -1], q[1, -1], color="red", s=40, label="End")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Body-Fixed x-y Trajectory")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(solution.t / 3600.0, radius)
    axes[1].set_xlabel("Time [hr]")
    axes[1].set_ylabel("Radius [m]")
    axes[1].set_title("Distance from Itokawa Center")
    axes[1].grid(True)

    fig.tight_layout()
    output_file = OUTPUT_DIR / "initial_orbit.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file


def compute_diagnostics(solution, gravity_model):
    q = solution.y[0:3, :]
    qdot = solution.y[3:6, :]

    energy = np.zeros(solution.t.size)
    angular_momentum = np.zeros(solution.t.size)

    for i in range(solution.t.size):
        position = q[:, i]
        velocity_rotating = qdot[:, i]
        velocity_inertial = velocity_rotating + np.cross(OMEGA_VEC, position)

        potential, _, _ = gravity_model(position, parallel=False)
        spin_velocity = np.cross(OMEGA_VEC, position)

        energy[i] = (
            0.5 * np.dot(velocity_rotating, velocity_rotating)
            - 0.5 * np.dot(spin_velocity, spin_velocity)
            - potential
        )
        angular_momentum[i] = np.linalg.norm(np.cross(position, velocity_inertial))

    return energy, angular_momentum


def plot_diagnostics(solution, gravity_model):
    OUTPUT_DIR.mkdir(exist_ok=True)

    energy, angular_momentum = compute_diagnostics(solution, gravity_model)
    time_hr = solution.t / 3600.0

    energy_drift = energy - energy[0]
    angular_momentum_change = angular_momentum - angular_momentum[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(time_hr, energy_drift)
    axes[0].set_xlabel("Time [hr]")
    axes[0].set_ylabel(r"$E - E_0$ [m$^2$/s$^2$]")
    axes[0].set_title("Rotating-Frame Energy Check")
    axes[0].grid(True)

    axes[1].plot(time_hr, angular_momentum_change)
    axes[1].set_xlabel("Time [hr]")
    axes[1].set_ylabel(r"$|h| - |h_0|$ [m$^2$/s]")
    axes[1].set_title("Angular Momentum Diagnostic")
    axes[1].grid(True)

    fig.tight_layout()
    output_file = OUTPUT_DIR / "conservation_checks.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file, energy_drift, angular_momentum_change


def main():
    gravity_model = load_gravity_model(MESH_FILE)

    initial_state = np.hstack((INITIAL_POSITION_M, INITIAL_VELOCITY_M_S))
    initial_derivative = lagrangian_eom(0.0, initial_state, gravity_model)
    initial_acceleration = initial_derivative[3:6]

    solution = propagate_orbit(gravity_model,INITIAL_POSITION_M,INITIAL_VELOCITY_M_S,)

    orbit_file = plot_orbit(solution)
    diagnostics_file, energy_drift, angular_momentum_change = plot_diagnostics(solution,gravity_model)

    printc = False
    if printc == True:
        print("Initial position [m]:", INITIAL_POSITION_M)
        print("Initial velocity [m/s]:", INITIAL_VELOCITY_M_S)
        print("Initial acceleration [m/s^2]:", initial_acceleration)
        print("Final position [m]:", solution.y[0:3, -1])
        print("Final velocity [m/s]:", solution.y[3:6, -1])
        print("Max rotating-frame energy drift:", np.max(np.abs(energy_drift)))
        print("Max angular momentum change:", np.max(np.abs(angular_momentum_change)))
        print("Saved orbit plot:", orbit_file)
        print("Saved diagnostics plot:", diagnostics_file)
    

    # Check symplecticity using a short-arc finite-difference STM in Hamiltonian coordinates.
    xi = lagrangian_to_hamiltonian(initial_state)

    stm = np.zeros((6, 6))
    for i in range(6):
        delta = 1e-3 if i < 3 else 1e-6

        xi_plus = np.copy(xi)
        xi_minus = np.copy(xi)
        xi_plus[i] += delta
        xi_minus[i] -= delta

        initial_state_plus = hamiltonian_to_lagrangian(xi_plus)
        initial_state_minus = hamiltonian_to_lagrangian(xi_minus)

        solution_plus = propagate_orbit_for_stm(
            gravity_model,
            initial_state_plus,
        )
        solution_minus = propagate_orbit_for_stm(
            gravity_model,
            initial_state_minus,
        )

        xf_plus = lagrangian_to_hamiltonian(solution_plus.y[:, -1])
        xf_minus = lagrangian_to_hamiltonian(solution_minus.y[:, -1])
        stm[:, i] = (xf_plus - xf_minus) / (2.0 * delta)

    J = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    symplectic_condition = stm.T @ J @ stm - J
    print(symplectic_condition)
    print("Norm of symplectic condition:", np.linalg.norm(symplectic_condition))


if __name__ == "__main__":
    main()
