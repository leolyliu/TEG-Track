import copy
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_duplicate_id(arr):
    """
    Return: list of tuples like (1,[0,1,2]), meaning that index 0,1,2 all have value of 1
    """

    uniq = np.unique(arr).tolist()
    ret = []

    for _ in range(len(uniq)):
        ret.append([])
    for index, nums in enumerate(arr):
        id = uniq.index(nums)
        ret[id].append(index)

    ans = [(uniq[i], ret[i]) for i in range(len(uniq))]
    return ans


def get_mapping(prev_markers, markers, max_distance=30):
    """
    Match markers between two adjacent frames.
    """
    
    prev_markers = copy.deepcopy(prev_markers)
    markers = copy.deepcopy(markers)

    prev_markers_array = np.array(prev_markers)
    markers = np.array(markers)
    if prev_markers_array.ndim <= 1:
        return np.zeros((markers.shape[0],)).astype(np.int), np.ones((markers.shape[0],)).astype(np.bool),
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(prev_markers_array)
    distances, indices = nbrs.kneighbors(markers)
    lost = distances > max_distance
    distances = distances.flatten()
    mapping = indices.flatten()
    lost = lost.flatten()

    dup = get_duplicate_id(mapping)
    for (value, index_list) in dup:
        min_id = np.argmin(distances[index_list])
        for duplicated in index_list:
            if duplicated != index_list[min_id]:
                lost[duplicated] = True

    return mapping, lost


def get_tactile_points(depth, model_matrix, Pixmm=5.9259259259e-5, offset=-0.006, threshold=0.0002, depth_scale=100000, dy=-0.025):
    """
    For the left/right tactile sensor, compute the tactile depth point cloud. Points are defined in the world coordinate system.
    Inputs:
        * depth: the tactile depth image in the current frame, shape = (320, 320)
        * model_matrix: the pad pose in the current frame, shape = (4, 4)
    Return:
        * world_points: The tactile depth point cloud in the world coordinate system, shape = (N_point, 3)
    """
    
    depth = depth / depth_scale
    H, W = depth.shape
    flag = (threshold < depth) & (depth < 0.049)
    pY, pX = np.where(flag)
    X = (W / 2 - pX) * Pixmm
    Y = (pY - H / 2) * Pixmm
    Z = depth[flag] + offset
    camera_points = np.vstack((X, Y, Z)).T
    rectified_camera_points = camera_points.copy()
    rectified_camera_points[:, 1] += dy
    world_points = np.matmul(rectified_camera_points, model_matrix[:3, :3].T) + model_matrix[:3, 3]
    return world_points


def get_contact(prev_marker_pts, marker_pts, dt, depth, last_depth, model_matrix, last_model_matrix, Pixmm=5.9259259259e-5, offset=-0.006, threshold=0.0002, depth_scale=100000, dy=-0.025):
    """
    For the left/right tactile sensor, compute 3D positions and velocities of contact points. Positions and velocities are defined in the world coordinate system.
    Inputs:
        * prev_marker_pts: a numpy array indicating marker pixels in the last frame, shape = (N_marker_0, 2)
        * marker_pts: a numpy array indicating marker pixels in the current frame, shape = (N_marker_1, 2)
        * dt: time difference between the last frame and the current frame. The unit is second (s)
        * depth: the tactile depth image in the current frame, shape = (320, 320)
        * last_depth: the tactile depth image in the last frame, shape = (320, 320)
        * model_matrix: the pad pose in the current frame, shape = (4, 4)
        * last_model_matrix: the pad pose in the last frame, shape = (4, 4)
    Returns:
        * contact_points: a numpy array indicating 3D marker positions relative to the world coordinate system, shape = (N_contact_point, 3)
        * contact_velocities: a numpy array indicating 3D marker velocities relative to the world coordinate system, shape = (N_contact_point, 3)
    """
    mapping_raw, lost_raw = get_mapping(prev_marker_pts, marker_pts)
    depth = depth / depth_scale
    last_depth = last_depth / depth_scale
    H, W = depth.shape
    contact_points = []
    contact_velocities = []
    for idx, point in enumerate(marker_pts):
        if depth[(int(point[1]), int(point[0]))] > threshold and lost_raw[idx] == False:
            camera_point = np.array(((W / 2 - point[0]) * Pixmm, (point[1] - H / 2) * Pixmm + dy, depth[(int(point[1]), int(point[0]))] + offset))
            last_point = prev_marker_pts[mapping_raw[idx]]
            last_camera_point = np.array(((W / 2 - last_point[0]) * Pixmm, (last_point[1] - H / 2) * Pixmm + dy, last_depth[(int(last_point[1]), int(last_point[0]))] + offset))
            world_point = np.matmul(camera_point, model_matrix[:3, :3].T) + model_matrix[:3, 3]
            contact_points.append(world_point)
            last_world_point = np.matmul(last_camera_point, last_model_matrix[:3, :3].T) + last_model_matrix[:3, 3]
            mean_velocity = (world_point - last_world_point) / dt
            contact_velocities.append(mean_velocity)
    return contact_points, contact_velocities
