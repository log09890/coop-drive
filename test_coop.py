# python test_coop.py  (numpy only, no CARLA needed)
import numpy as np
from simulation.coop import sector_distances

R, N = 50.0, 8
args = dict(ego_half_extent=(2.5, 1.2), n_sectors=N, max_range=R, z_min=-2.2, z_max=-0.5)
ego = (0, 0, 2.5, 0, 0, 0)
front = N // 2  # sector 0 starts behind (angle -pi), so +x is in the middle

# ego's own scan: obstacle 10 m ahead, ground point, zero padding -> only the obstacle counts
d = sector_distances([[[10, 0, -1.5, 1], [5, 0, -2.5, 1], [0, 0, 0, 0]]], [ego], ego, **args)
assert np.isclose(d[front], 10 / R) and np.all(np.delete(d, front) == 1), d

# coop car 20 m ahead facing us (yaw 180) sees an obstacle 5 m in front of it -> 15 m ahead of ego
d = sector_distances([[[5, 0, -1.5, 1]]], [(20, 0, 2.5, 0, 180, 0)], ego, **args)
assert np.isclose(d[front], 15 / R), d

# coop car sees the ego body itself -> ignored; ego turned 90 deg -> obstacle on world +y is now straight ahead
d = sector_distances([[[20, 0, -1.5, 1]], [[0, 8, -1.5, 1]]], [(20, 0, 2.5, 0, 180, 0), (0, 0, 2.5, 0, 0, 0)],
                     (0, 0, 2.5, 0, 90, 0), **args)
assert np.isclose(d[front], 8 / R) and np.isclose(d.min(), 8 / R), d
print('ok')

from simulation.coop import corridor_distance
straight = [(x, 0, 0) for x in range(3, 21)]  # road surface at world z 0, lidar 2.5 m above it
c = dict(half_width=1.3, z_min=0.3, z_max=1.5, min_points=1)
# roadside crop at 8 m but 3 m to the side -> clear; pedestrian 12 m straight ahead seen by a coop car -> 9 m along the path
assert corridor_distance([[[8, 3, -1.5, 1]]], [ego], straight, **c) == np.inf
assert np.isclose(corridor_distance([[[8, 3, -1.5, 1]], [[8, 0, -1.5, 1]]], [ego, (20, 0, 2.5, 0, 180, 0)], straight, **c), 9)
# curve to the right: crop straight ahead at 10 m is off the lane, car in the curve is on it
curve = [(3 + 8 * np.sin(t), 8 * (1 - np.cos(t)), 0) for t in np.linspace(0, 1.2, 15)]
assert corridor_distance([[[10, 0, -1.5, 1]]], [ego], curve, **c) == np.inf
assert corridor_distance([[[curve[8][0], curve[8][1], -1.5, 1]]], [ego], curve, **c) < np.inf
# uphill road: the road surface itself is not an obstacle
hill = [(x, 0, 0.15 * x) for x in range(3, 21)]
assert corridor_distance([[[15, 0, -2.5 + 0.15 * 15, 1]]], [ego], hill, **c) == np.inf
# coop car behind sees the ego hood (1.5 m ahead of the ego lidar) -> ignored with the ego box
hood = [[[11.5, 0, -1.5, 1]]]
assert corridor_distance(hood, [(-10, 0, 2.5, 0, 0, 0)], [(x, 0, 0) for x in range(0, 21)], **c) < np.inf
assert corridor_distance(hood, [(-10, 0, 2.5, 0, 0, 0)], [(x, 0, 0) for x in range(0, 21)], **c, ego_pose=ego, ego_half_extent=(2.7, 1.3)) == np.inf
# a lone overhanging point does not count with the default min_points=3; a real object (many points) does
three = dict(half_width=1.3, z_min=0.3, z_max=1.5)
assert corridor_distance([[[10, 0, -1.4, 1]]], [ego], straight, **three) == np.inf
assert corridor_distance([[[10, 0, -1.4, 1], [10, 0.2, -1.6, 1], [10.5, 0, -1.8, 1]]], [ego], straight, **three) == 7
print('corridor ok')
