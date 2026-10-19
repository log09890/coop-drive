'''
Cooperative perception (idea, lidar setup and pose format from CODD:
https://github.com/eduardohenriquearnold/CODD).

Every cooperative car shares its lidar scan + sensor pose. We fuse all scans in the
ego lidar frame and squash them into "nearest obstacle per angular sector", a small
vector the PPO agent can consume next to the VAE latent.

Numpy only, so it also runs offline on CODD .hdf5 snippets (pointcloud + lidar_pose).
'''
import numpy as np


def pose_matrix(x, y, z, pitch, yaw, roll):
    '''Local -> global 4x4 matrix, UE4 left-handed, degrees (same as CODD vis.getTransform).'''
    pitch, yaw, roll = np.radians([pitch, yaw, roll])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
                     [sp, -cp * sr, cp * cr, z],
                     [0, 0, 0, 1]])


def to_ego(pts, pose, ego_inv):
    '''[N, >=3] points in a sensor frame -> [M, 3] in the ego lidar frame (x fwd, y right, z up); drops CODD zero padding.'''
    pts = np.asarray(pts, dtype=np.float64)[:, :3]
    pts = pts[np.any(pts != 0, axis=1)]
    return (np.c_[pts, np.ones(len(pts))] @ (ego_inv @ pose_matrix(*pose)).T)[:, :3]


def sector_distances(clouds, poses, ego_pose, ego_half_extent, n_sectors, max_range, z_min, z_max):
    '''
    clouds: list of [N, >=3] points, each in its own sensor frame.
    poses:  list of (x, y, z, pitch, yaw, roll) sensor poses, same order as clouds.
    ego_pose: pose of the ego lidar. ego_half_extent: (x, y) half size of the ego car.
    z_min/z_max: height band kept, in ego lidar frame (drops ground and overhead points).
    Returns [n_sectors] in [0, 1]: nearest obstacle / max_range, 1 = free. Sector 0 starts behind the car.
    '''
    ego_inv = np.linalg.inv(pose_matrix(*ego_pose))
    out = np.full(n_sectors, float(max_range))
    ex, ey = ego_half_extent
    for pts, pose in zip(clouds, poses):
        x, y, z = to_ego(pts, pose, ego_inv).T
        keep = (z > z_min) & (z < z_max) & ~((np.abs(x) < ex) & (np.abs(y) < ey))  # other cars see the ego body
        x, y = x[keep], y[keep]
        idx = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
        np.minimum.at(out, idx, np.hypot(x, y))
    return np.minimum(out, max_range) / max_range


def corridor_distance(clouds, poses, path, half_width, z_min, z_max, ego_pose=None, ego_half_extent=None, min_points=3):
    '''
    Nearest obstacle along the lane ahead: points within half_width of the route path and z_min..z_max above its road surface.
    path: [K, 3] route waypoints ahead (world frame, on the road, ~1 m apart, starting at the front bumper).
    ego_pose/ego_half_extent: drop points on the ego body (other cars' lidars see it).
    min_points: hits needed to count as an obstacle (single points from overhanging branches stopped the car for good).
    Returns the distance along the path in metres, inf if the lane is clear. Following the route keeps roadside
    crops in curves and the road itself on slopes out of the corridor.
    '''
    path = np.asarray(path, dtype=np.float64)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    hits = []
    for pts, pose in zip(clouds, poses):
        p = to_ego(pts, pose, np.eye(4))  # identity: straight to world frame
        d = np.linalg.norm(p[:, None, :2] - path[None, :, :2], axis=2)  # [N, K]
        k = d.argmin(1)
        h = p[:, 2] - path[k, 2]
        m = (d[np.arange(len(p)), k] < half_width) & (h > z_min) & (h < z_max) & (k < len(path) - 1)  # past the end: not ours
        if ego_pose is not None and m.any():
            x, y, _ = to_ego(p[m], (0, 0, 0, 0, 0, 0), np.linalg.inv(pose_matrix(*ego_pose))).T
            body = (np.abs(x) < ego_half_extent[0]) & (np.abs(y) < ego_half_extent[1])
            m[np.flatnonzero(m)[body]] = False
        hits.append(along[k[m]])
    hits = np.sort(np.concatenate(hits)) if hits else np.zeros(0)
    return hits[min_points - 1] if len(hits) >= min_points else np.inf
