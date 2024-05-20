import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import time
from scipy import interpolate
import open3d


def load_model(model_path, K_points=100000, scale=1.0):
    mesh = o3d.io.read_triangle_mesh(model_path)
    pcd = mesh.sample_points_uniformly(number_of_points=K_points)
    points = np.array(pcd.points) * scale
    return points


def load_completed_pcd(file_path):
    pcd = open3d.io.read_point_cloud(file_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    return np.array(pcd.points)


def load_model_dense_pcd(model_path, K_points=100000, scale=1.0):
    mesh = o3d.io.read_triangle_mesh(model_path)
    pcd = mesh.sample_points_uniformly(number_of_points=K_points)
    points = np.array(pcd.points) * scale
    tree = KDTree(points)
    return tree


def calc_distance_KDTree(tree, query):
    min_dists = tree.query(query, 1)[0].reshape(-1)
    return min_dists


def sdf_grid(model_path, K_points=100000, sample_grid=(100,100,100)):
    tree = load_model_dense_pcd(model_path, K_points)
    x, y, z = sample_grid
    i = np.linspace(-0.5, 0.5, x).reshape(-1)
    j = np.linspace(-0.5, 0.5, y).reshape(-1)
    k = np.linspace(-0.5, 0.5, z).reshape(-1)
    xx, yy, zz =np.meshgrid(i, j, k)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    zz = np.reshape(zz, (1, -1))
    grid = np.concatenate([xx, yy, zz], axis=0).T
    dist = calc_distance_KDTree(tree, grid)
    sdf_grid = np.zeros((x, y, z))
    idx = 0
    for a in range(x):
        for b in range(y):
            for c in range(z):
                sdf_grid[a][b][c] = dist[idx]
                idx += 1
    grid = (i.T, j.T, k.T)
    return grid, sdf_grid


def pcd_denoise_and_downsample(visual_points, tactile_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visual_points)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    visual_points = np.array(pcd.points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tactile_points)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    tactile_points = np.array(pcd.points)
    all_points = np.concatenate((visual_points, tactile_points), axis=0)

    return visual_points, tactile_points, all_points


def sdf_any(pts, grid, sdf_grid):
    interpolating_func = interpolate.RegularGridInterpolator(grid, sdf_grid, bounds_error=False)
    sdf = interpolating_func(pts)
    return sdf


if __name__ == "__main__":
    model_path = "/nas/datasets/Tactile-tracking_Data/ShapeNetCore.v1/02946921/fe6be0860c63aa1d8b2bf9f4ef8234/model.obj"
    tree = load_model_dense_pcd(model_path, K_points=100000)
    query = np.random.randn(50000, 3)
    start_time = time.time()
    dist = calc_distance_KDTree(tree, query)
    print(time.time() - start_time)
