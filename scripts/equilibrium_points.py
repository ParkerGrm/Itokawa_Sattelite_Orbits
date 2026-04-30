import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from scipy.optimize import root

from eom import equilibrium_residual, lagrangian_stability_matrix, load_gravity_model


MESH_FILE = PROJECT_ROOT / "data" / "itokawa_50k_ascii.ply"
OUTPUT_DIR = PROJECT_ROOT / "results"
SEARCH_LIMIT_M = 1500.0


def initial_guesses():
    guesses = []
    radii = [450.0, 500.0, 600.0]
    directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]

    for radius in radii:
        for direction in directions:
            direction = np.asarray(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            guesses.append(radius * direction)

    return guesses


def is_new_point(point, points, tolerance=1.0):
    for existing_point in points:
        if np.linalg.norm(point - existing_point) < tolerance:
            return False
    return True


def bounded_residual(q, gravity_model):
    if np.linalg.norm(q) > SEARCH_LIMIT_M:
        return np.ones(3) * 1e3

    return equilibrium_residual(q, gravity_model)


def find_equilibrium_points(gravity_model):
    points = []

    for guess in initial_guesses():
        result = root(lambda q: bounded_residual(q, gravity_model), guess)

        residual_norm = np.linalg.norm(equilibrium_residual(result.x, gravity_model))
        if result.success and np.linalg.norm(result.x) < SEARCH_LIMIT_M and residual_norm < 1e-8:
            if is_new_point(result.x, points):
                points.append(result.x)

    return points


def classify_stability(eigenvalues, tolerance=1e-8):
    max_real_part = np.max(np.real(eigenvalues))

    if max_real_part > tolerance:
        return "unstable"

    return "linearly stable"


def analyze_points(points, gravity_model):
    rows = []

    for i, point in enumerate(points, start=1):
        matrix = lagrangian_stability_matrix(point, gravity_model)
        eigenvalues = np.linalg.eigvals(matrix)
        rows.append(
            {
                "name": f"E{i}",
                "point": point,
                "residual_norm": np.linalg.norm(equilibrium_residual(point, gravity_model)),
                "eigenvalues": eigenvalues,
                "max_real_part": np.max(np.real(eigenvalues)),
                "classification": classify_stability(eigenvalues),
            }
        )

    return rows


def plot_equilibrium_points(rows):
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

    for row in rows:
        point = row["point"]
        ax.scatter(point[0], point[1], s=45)
        ax.text(point[0], point[1], f" {row['name']}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Computed Equilibrium Points")
    ax.grid(True)
    ax.legend()

    output_file = OUTPUT_DIR / "equilibrium_points.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file


def print_summary(rows):
    print("Name        x [m]        y [m]        z [m]     residual      max Re(lambda)   class")
    print("-" * 96)

    for row in rows:
        point = row["point"]
        print(
            f"{row['name']:>4s}"
            f"{point[0]:13.3f}"
            f"{point[1]:13.3f}"
            f"{point[2]:13.3f}"
            f"{row['residual_norm']:13.3e}"
            f"{row['max_real_part']:16.3e}"
            f"   {row['classification']}"
        )

    print("\nEigenvalues:")
    for row in rows:
        print(row["name"])
        for value in row["eigenvalues"]:
            print(f"    {value.real: .6e} {value.imag:+.6e}j")


def main():
    gravity_model = load_gravity_model(MESH_FILE)
    points = find_equilibrium_points(gravity_model)
    rows = analyze_points(points, gravity_model)
    output_file = plot_equilibrium_points(rows)

    print_summary(rows)
    print("\nSaved plot:", output_file)


if __name__ == "__main__":
    main()
