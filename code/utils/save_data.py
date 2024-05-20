import numpy as np
import pickle
from transforms3d.euler import euler2mat


def save_tracking_results(states, save_path, save_dynamics=True, save_constants=True, constants=None):
    translations = []
    rotations = []
    if save_dynamics:
        velocities = []
        angular_velocities = []

    for state in states:
        t = state[0:3]
        r = euler2mat(state[3], state[4], state[5])
        translations.append(t)
        rotations.append(r)
        if save_dynamics:
            v = state[6:9]
            omega = state[9:12]
            velocities.append(v)
            angular_velocities.append(omega)
    
    translations = np.array(translations)  # shape = (N_frame, 3)
    rotations = np.array(rotations)  # shape = (N_frame, 3, 3)
    if save_dynamics:
        velocities = np.array(velocities)  # shape = (N_frame, 3)
        angular_velocities = np.array(angular_velocities)  # shape = (N_frame, 3)

    results = {"translations": translations, "rotations": rotations}
    if save_dynamics:
        results["velocities"] = velocities
        results["angular_velocities"] = angular_velocities

    if save_constants:
        results["mass"] = constants["mass"]
        results["inertia"] = constants["inertia"]
    pickle.dump(results, open(save_path, "wb"))
