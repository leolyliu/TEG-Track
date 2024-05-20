import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_affine_transformation(contact_pts, last_pts):
    contact_pts = contact_pts.transpose(1, 0)
    last_pts = last_pts.transpose(1, 0)
    
    # matching
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(last_pts)
    distances, indices = nbrs.kneighbors(contact_pts)
    distances = distances.squeeze(axis=1)
    indices = indices.squeeze(axis=1)
    valid_matches = distances <= 30
    x = contact_pts[valid_matches]
    y = last_pts[indices][valid_matches]

    # least-square
    A = np.zeros((x.shape[0] * 2, 6))
    A[:x.shape[0], 0:2] = x
    A[:x.shape[0], 2] = 1
    A[x.shape[0]:, 3:5] = x
    A[x.shape[0]:, 5] = 1
    b = y.transpose(1, 0).reshape(-1)
    affine_trans = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.linalg.norm(A @ affine_trans - b, ord=2)
