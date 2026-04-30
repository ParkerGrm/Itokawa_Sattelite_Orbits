import numpy as np
import pyvista as pv
from polyhedral_gravity import (
    GravityEvaluable,
    MetricUnit,
    Polyhedron,
    PolyhedronIntegrity,
)

## EOM for a particle in the rotating frame of an irregularly shaped asteroid. ##

# Itokawa physical parameters
DENSITY_KG_M3 = 1.673e3
ROTATION_PERIOD_S = 12.1324 * 3600.0
OMEGA = 2.0 * np.pi / ROTATION_PERIOD_S
OMEGA_VEC = np.array([0.0, 0.0, OMEGA])

# The downloaded Itokawa mesh coordinates are in km. The EOM use SI units.
MESH_SCALE_TO_METERS = 1000.0


def load_gravity_model(mesh_file):
    """Create a cached polyhedron gravity model from an Itokawa PLY mesh."""
    mesh = pv.read(mesh_file)

    vertices_m = np.asarray(mesh.points, dtype=float) * MESH_SCALE_TO_METERS
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)

    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected the mesh to contain only triangular faces.")

    polyhedron = Polyhedron(
        polyhedral_source=(vertices_m, faces[:, 1:]),
        density=DENSITY_KG_M3,
        metric_unit=MetricUnit.METER,
        integrity_check=PolyhedronIntegrity.DISABLE,
    )

    return GravityEvaluable(polyhedron=polyhedron)


def gravity_acceleration(gravity_model, q):
    """Return the polyhedron gravitational acceleration at body-fixed position q."""
    potential, acceleration, tensor = gravity_model(q, parallel=False)
    return np.asarray(acceleration, dtype=float)


def gravity_hessian(gravity_model, q):
    """Return the gravity-gradient tensor d(grad U)/dq at body-fixed position q."""
    potential, acceleration, tensor = gravity_model(q, parallel=False)
    uxx, uyy, uzz, uxy, uxz, uyz = np.asarray(tensor, dtype=float)
    return np.array(
        [
            [uxx, uxy, uxz],
            [uxy, uyy, uyz],
            [uxz, uyz, uzz],
        ]
    )


def skew(vector):
    """Return the matrix form of the cross product by vector."""
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def lagrangian_to_hamiltonian(state_l):
    """Convert [q, qdot] to [q, p]."""
    q = state_l[0:3]
    qdot = state_l[3:6]
    p = qdot + np.cross(OMEGA_VEC, q)
    return np.hstack((q, p))


def hamiltonian_to_lagrangian(state_h):
    """Convert [q, p] to [q, qdot]."""
    q = state_h[0:3]
    p = state_h[3:6]
    qdot = p - np.cross(OMEGA_VEC, q)
    return np.hstack((q, qdot))


def lagrangian_eom(t, state_l, gravity_model):
    """Rotating-frame EOM for the Lagrangian state [q, qdot]."""
    q = state_l[0:3]
    qdot = state_l[3:6]

    gravity = gravity_acceleration(gravity_model, q)
    coriolis = -2.0 * np.cross(OMEGA_VEC, qdot)
    centrifugal = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, q))
    qddot = gravity + coriolis + centrifugal

    return np.hstack((qdot, qddot))


def hamiltonian_eom(t, state_h, gravity_model):
    """Rotating-frame EOM for the Hamiltonian state [q, p]."""
    q = state_h[0:3]
    p = state_h[3:6]

    gravity = gravity_acceleration(gravity_model, q)
    qdot = p - np.cross(OMEGA_VEC, q)
    pdot = gravity - np.cross(OMEGA_VEC, p)

    return np.hstack((qdot, pdot))


def equilibrium_residual(q, gravity_model):
    """Zero at a stationary point in the rotating frame."""
    gravity = gravity_acceleration(gravity_model, q)
    centrifugal = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, q))
    return gravity + centrifugal


def lagrangian_stability_matrix(q_eq, gravity_model):
    """Linearized [q, qdot] dynamics matrix at an equilibrium point."""
    omega_cross = skew(OMEGA_VEC)
    u_q_q = gravity_hessian(gravity_model, q_eq)

    top = np.hstack((np.zeros((3, 3)), np.eye(3)))
    bottom = np.hstack((u_q_q - omega_cross @ omega_cross, -2.0 * omega_cross))
    return np.vstack((top, bottom))
