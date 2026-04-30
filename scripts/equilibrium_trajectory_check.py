import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_FILE = PROJECT_ROOT / "data" / "itokawa_50k_ascii.ply"
OUTPUT_DIR = PROJECT_ROOT / "results"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("POLYHEDRAL_GRAVITY_LOGGING_LEVEL", "ERROR")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import root

from eom import ROTATION_PERIOD_S, equilibrium_residual, lagrangian_eom, load_gravity_model


EQUILIBRIUM_POINTS = {
    "E1": np.array([497.09540099, -3.85061979, -9.82762765]),
    "E2": np.array([-484.40412252, 50.45054814, -5.02376231]),
    "E3": np.array([54.86623751, 438.36274394, 2.11546248]),
    "E4": np.array([12.98461457, -444.72519203, 1.38155562]),
}

SHORT_TIME_SPAN_S = (0.0, 0.25 * ROTATION_PERIOD_S)
LONG_TIME_SPAN_S = (0.0, 3.0 * ROTATION_PERIOD_S)
NUM_OUTPUT_POINTS = 500


def refine_equilibrium_point(gravity_model, equilibrium_point):
    result = root(lambda q: equilibrium_residual(q, gravity_model), equilibrium_point)

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


def propagate_from_equilibrium(gravity_model, equilibrium_point, time_span_s):
    state0 = np.hstack((equilibrium_point, np.zeros(3)))
    output_times = np.linspace(time_span_s[0], time_span_s[1], NUM_OUTPUT_POINTS)

    return solve_ivp(
        fun=lambda t, state: lagrangian_eom(t, state, gravity_model),
        t_span=time_span_s,
        y0=state0,
        method="DOP853",
        t_eval=output_times,
        rtol=1e-11,
        atol=1e-13,
    )


def plot_displacement(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, equilibrium_point, solution in results:
        displacement = solution.y[0:3, :].T - equilibrium_point
        displacement_norm = np.linalg.norm(displacement, axis=1)

        axes[0].plot(
            solution.t / 3600.0,
            displacement_norm,
            label=name,
        )
        axes[1].plot(
            displacement[:, 0],
            displacement[:, 1],
            label=name,
        )

    axes[0].set_xlabel("Time [hr]")
    axes[0].set_ylabel(r"$\|\mathbf{q}(t)-\mathbf{q}_{eq}\|$ [m]")
    axes[0].set_title("Equilibrium Trajectory Drift")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel(r"$\Delta x$ [m]")
    axes[1].set_ylabel(r"$\Delta y$ [m]")
    axes[1].set_title("x-y Drift from Equilibrium")
    axes[1].grid(True)
    axes[1].axis("equal")
    axes[1].legend()

    fig.tight_layout()
    output_file = OUTPUT_DIR / "equilibrium_trajectory_check.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file


def plot_trajectories_near_equilibrium_points(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh = pv.read(MESH_FILE)
    asteroid_points_m = np.asarray(mesh.points) * 1000.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        asteroid_points_m[::30, 0],
        asteroid_points_m[::30, 1],
        s=1,
        color="0.7",
        label="Itokawa shape projection",
    )

    for name, equilibrium_point, solution in results:
        trajectory = solution.y[0:3, :].T

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=1.5,
            label=f"{name} trajectory",
        )
        ax.scatter(equilibrium_point[0], equilibrium_point[1], s=45)
        ax.text(equilibrium_point[0], equilibrium_point[1], f" {name}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectories Initialized at Equilibrium Points")
    ax.grid(True)
    ax.legend()

    output_file = OUTPUT_DIR / "equilibrium_trajectories_xy.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file


def set_equal_3d_axes(ax, points):
    x_limits = [np.min(points[:, 0]), np.max(points[:, 0])]
    y_limits = [np.min(points[:, 1]), np.max(points[:, 1])]
    z_limits = [np.min(points[:, 2]), np.max(points[:, 2])]

    x_center = 0.5 * (x_limits[0] + x_limits[1])
    y_center = 0.5 * (y_limits[0] + y_limits[1])
    z_center = 0.5 * (z_limits[0] + z_limits[1])
    radius = 0.5 * max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )

    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
    ax.set_zlim(z_center - radius, z_center + radius)


def plot_trajectories_near_equilibrium_points_3d(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh = pv.read(MESH_FILE)
    asteroid_points_m = np.asarray(mesh.points) * 1000.0

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        asteroid_points_m[::30, 0],
        asteroid_points_m[::30, 1],
        asteroid_points_m[::30, 2],
        s=1,
        color="0.7",
        alpha=0.35,
        label="Itokawa shape",
    )

    axis_points = [asteroid_points_m]

    for name, equilibrium_point, solution in results:
        trajectory = solution.y[0:3, :].T
        axis_points.append(trajectory)
        axis_points.append(equilibrium_point.reshape(1, 3))

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linewidth=1.5,
            label=f"{name} trajectory",
        )
        ax.scatter(
            equilibrium_point[0],
            equilibrium_point[1],
            equilibrium_point[2],
            s=45,
        )
        ax.text(
            equilibrium_point[0],
            equilibrium_point[1],
            equilibrium_point[2],
            f" {name}",
        )

    all_points = np.vstack(axis_points)
    set_equal_3d_axes(ax, all_points)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Trajectories Initialized at Equilibrium Points")
    ax.view_init(elev=25, azim=35)
    ax.legend()

    output_file = OUTPUT_DIR / "equilibrium_trajectories_3d.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file


def main():
    gravity_model = load_gravity_model(MESH_FILE)
    short_results = []
    long_results = []

    for name, point in EQUILIBRIUM_POINTS.items():
        refined_point = refine_equilibrium_point(gravity_model, point)

        short_solution = propagate_from_equilibrium(
            gravity_model, refined_point, SHORT_TIME_SPAN_S
        )
        long_solution = propagate_from_equilibrium(
            gravity_model, refined_point, LONG_TIME_SPAN_S
        )

        if not short_solution.success:
            raise RuntimeError(short_solution.message)
        if not long_solution.success:
            raise RuntimeError(long_solution.message)

        short_displacement = short_solution.y[0:3, -1] - refined_point
        long_displacement = long_solution.y[0:3, -1] - refined_point
        residual = equilibrium_residual(refined_point, gravity_model)

        short_results.append((name, refined_point, short_solution))
        long_results.append((name, refined_point, long_solution))

        print(name)
        print("  residual norm [m/s^2]:", np.linalg.norm(residual))
        print("  refined point [m]:", refined_point)
        print("  short final displacement [m]:", short_displacement)
        print("  short final displacement norm [m]:", np.linalg.norm(short_displacement))
        print("  long final displacement [m]:", long_displacement)
        print("  long final displacement norm [m]:", np.linalg.norm(long_displacement))

    displacement_plot = plot_displacement(long_results)
    trajectory_plot = plot_trajectories_near_equilibrium_points(long_results)
    trajectory_3d_plot = plot_trajectories_near_equilibrium_points_3d(long_results)

    print("Saved plot:", displacement_plot)
    print("Saved plot:", trajectory_plot)
    print("Saved plot:", trajectory_3d_plot)


if __name__ == "__main__":
    main()
