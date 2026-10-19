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

from simulation.coop import safety_penalty
assert safety_penalty(np.inf, 10, 20, False, 0, 0) == 0  # calm driving costs nothing
assert np.isclose(safety_penalty(1.5, 10, 20, False, 0, 0), 0.25)  # near miss
assert safety_penalty(1.5, 0.5, 3, False, 0, 0) == 0  # crawling in a swarm is fine
assert np.isclose(safety_penalty(np.inf, 10, 20, True, 0, 0), 0.5)  # red light
assert np.isclose(safety_penalty(np.inf, 10, 20, False, 2, 2), 0.2)  # jerk is capped
print('safety ok')

from evaluate import summarize
s = summarize([dict(km=0.5, route_done=True, progress=1.0, collisions=1, near_miss_steps=20, red_runs=0, mean_speed=15, jerk=0.1, end='route_done'),
               dict(km=0.5, route_done=False, progress=0.5, collisions=0, near_miss_steps=0, red_runs=2, mean_speed=5, jerk=0.3, end='stuck')])
assert s['collisions_per_km'] == 1 and s['near_miss_s_per_km'] == 1.0 and s['red_light_runs_per_km'] == 2 and s['route_done_%'] == 50
print('summary ok')

import torch
from insight import closest_approach, input_sensitivity, group_sensitivity, SessionMetrics
# bike 10 m ahead coming straight at us at 5 m/s -> meets in 2 s at 0 m; bike moving away -> closest now
t, d = closest_approach([[10, 0], [10, 0]], [[-5, 0], [5, 0]])
assert np.allclose(t, [2, 0]) and np.allclose(d, [0, 10])
# crossing 1 m off our path still counts as a conflict at t = 2 s
t, d = closest_approach([[10, 1]], [[-5, 0]]); assert np.isclose(t[0], 2) and np.isclose(d[0], 1)
# sensitivity of a linear "actor": d out / d x = weights
lin = torch.nn.Linear(4, 2, bias=False)
with torch.no_grad(): lin.weight.copy_(torch.tensor([[1., 0, 0, -2], [0, 3, 0, 0]]))
st, pe = input_sensitivity(lin, torch.zeros(4)); assert np.allclose(st, [1, 0, 0, 2]) and np.allclose(pe, [0, 3, 0, 0])
g = group_sensitivity(np.ones(95 + 5 + 34), 95, 16)
assert g['camera (VAE)'] == 95 and g['LiDAR sectors'] == 16 and g['lane corridor'] == 1 and g['motion (deltas)'] == 17
m = SessionMetrics()
for _ in range(20): m.step(36.0, 1.5, False)  # 10 m/s for 1 s, all of it a near miss
assert np.isclose(m.km, 0.01) and np.isclose(m.near_miss_s, 1.0) and m.step(36.0, 9, True) == ['red_light'] and m.red_runs == 1
print('insight ok')

from simulation.infrastructure import V2XChannel, ObjectExtractor, encode_infra_obs, light_phase_left, INFRA_DIM, INFRA_K
ch = V2XChannel(latency=0.1, jitter=0.0, loss=0.0, range_m=100)
ch.send(0.0, (0, 0), {'t': 0.0, 'n': 1}); ch.send(0.0, (500, 0), {'t': 0.0, 'n': 2})
assert ch.receive(0.05, (0, 0)) == [] and [m['n'] for m in ch.receive(0.1, (0, 0))] == [1]  # on time, far one dropped
lossy = V2XChannel(loss=0.3, seed=1)
for i in range(2000): lossy.send(0.0, (0, 0), {'t': 0.0})
assert abs(len(lossy.receive(1.0, (0, 0))) / 2000 - 0.7) < 0.05
# RSU: a wall stays put -> learned as background; a bike moving 5 m/s along x -> one object with vx ~ 5
ex = ObjectExtractor(background_rate=0.05)
wall = np.array([[20, y, 1.0] for y in np.arange(-5, 5, 0.2)] + [[20.3, y, 1.5] for y in np.arange(-5, 5, 0.2)])
bike = lambda x: np.array([[x + dx, dy, z] for dx in (0, 0.7, 1.4, 2.0) for dy in (-0.2, 0.2) for z in (0.6, 1.2)])
for k in range(60):
    objs = ex(np.r_[wall, bike(k * 0.25)], 0.0, k * 0.05)  # 0.25 m per 0.05 s = 5 m/s
assert len(objs) == 1 and objs[0]['kind'] == 'two-wheeler' and abs(objs[0]['vx'] - 5) < 1.0, objs
o = encode_infra_obs((0, 0), 0.0, (0, 0), [{'x': 20, 'y': 0, 'vx': 0, 'vy': 0, 't': 0.0}, {'x': 5, 'y': 3, 'vx': -2, 'vy': 0, 't': 0.0}],
                     0.0, light=(25.0, 'Red', 6.0))
assert o.shape == (INFRA_DIM,) and np.allclose(o[:4], [5 / 30, 3 / 30, -0.2, 0]) and np.isclose(o[4], 20 / 30)  # nearest first
assert np.allclose(o[INFRA_K * 4:INFRA_K * 4 + 5], [0.5, 1, 0, 0, 0.2])
assert light_phase_left('Red', 3.0, 4.0, 1.5, 15.0) == 12.0
print('infrastructure ok')
from simulation.infrastructure import group_red_left
# 4-light group, 4 s green + 1.5 s yellow each; light 1 is green with 3 s (+1.5 yellow) left: light 3 waits for 1 and 2
assert group_red_left(3, 1, 4.5, [(4, 1.5)] * 4) == 4.5 + 5.5 and group_red_left(2, 1, 4.5, [(4, 1.5)] * 4) == 4.5
assert group_red_left(3, 1, 4.5, [(4, 1.5)] * 4, clearance=2) == 4.5 + 2 + 5.5 + 2  # + an all-red gap per turn
print('spat ok')

from simulation.infrastructure import edge_conflicts, dedupe
ego = (0, 0, 5, 0, 0.0)  # driving +x at 5 m/s
crossing = {'x': 10, 'y': -5, 'vx': 0, 'vy': 2.5}  # reaches our line at x = 10 in 2 s, when we are there
behind = {'x': -6, 'y': 0, 'vx': 6, 'vy': 0}  # catching up from behind: meets behind our nose -> no warning
beside = {'x': 0, 'y': 3, 'vx': 5, 'vy': 0}  # riding along 3 m to the side
hits = edge_conflicts(ego, [crossing, behind, beside])
assert len(hits) == 1 and hits[0][2] is crossing and abs(hits[0][0] - 2) < 0.05, hits
assert len(dedupe([{'x': 0, 'y': 0, 't': 1.0}, {'x': 0.8, 'y': 0, 't': 1.2}, {'x': 9, 'y': 0, 't': 1.0}])) == 2
ch = V2XChannel(latency=0.0, jitter=0.0, loss=0.0)
ch.send(0, (0, 0), {'t': 0, 'to': 'ego', 'type': 'DENM'}); ch.send(0, (0, 0), {'t': 0, 'type': 'CAM'})
box = ch.receive_all(0.0, {'ego': (0, 0), 'rsu1': (5, 0)})
assert len(box['ego']) == 2 and [m['type'] for m in box['rsu1']] == ['CAM']  # DENM is unicast
w = encode_infra_obs((0, 0), 0.0, (0, 0), [], 0.0, warning=1.5)
assert np.allclose(w[INFRA_K * 4 + 5:INFRA_K * 4 + 7], [1, 0.5])
print('edge ok')

from occlusion_eval import lead_time, summarize as occ_summary
assert lead_time(1.0, 5.0, 6.0) == 1.0 and lead_time(5.5, 5.0, 6.0) == 0.5 and lead_time(6.5, 5.0, 6.0) == -0.5 and lead_time(None, 5.0, 6.0) is None
s = occ_summary([{'lead': 1.0, 'aware_at_dart': True, 'collision': False, 'min_dist': 4}, {'lead': -0.5, 'aware_at_dart': False, 'collision': True, 'min_dist': 0.5},
                 {'lead': None, 'aware_at_dart': False, 'collision': True, 'min_dist': 0.2}])
assert s['aware_at_dart_%'] == 33.3 and s['aware_before_crossing_%'] == 33.3 and s['mean_lead_s'] == 0.25 and s['collision_%'] == 66.7
print('occlusion ok')
parked = {'x': 10, 'y': 0.5, 'vx': 0, 'vy': 0}  # parked right on our line: the corridor handles it, not a DENM
assert edge_conflicts((0, 0, 5, 0, 0.0), [parked]) == []
print('edge static ok')
# a bike that stops for 10 s stays an object (its cells are held out of the background), a wall still becomes background
ex = ObjectExtractor(background_rate=0.05)
for k in range(300):  # the bike arrives after the 3 s start-up, then waits 10 s
    objs = ex(np.r_[wall, bike(5.0)] if k >= 100 else wall, 0.0, k * 0.05)
assert len(objs) == 1 and objs[0]['kind'] == 'two-wheeler', objs
print('waiting bike ok')
# clutter present from the first frame becomes background even though it is small; a bike that arrives later and waits stays
ex = ObjectExtractor(background_rate=0.05)
pole = np.array([[0, 10, z] for z in (0.5, 1.0, 1.5, 2.0)] + [[0.1, 10, z] for z in (0.5, 1.0, 1.5, 2.0)])
for k in range(300):
    objs = ex(np.r_[wall, pole] if k < 100 else np.r_[wall, pole, bike(5.0)], 0.0, k * 0.05)
assert len(objs) == 1 and objs[0]['kind'] == 'two-wheeler', objs
print('clutter ok')
