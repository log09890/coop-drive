# python test_expert.py  (no CARLA: a fake env with a straight route)
from types import SimpleNamespace as NS
import numpy as np
from bc import expert_action


class Loc(NS):
    pass


def env(front, v, t, warn=None, red=False):
    tr = NS(location=Loc(x=0.0, y=0.0), get_forward_vector=lambda: Loc(x=1.0, y=0.0))
    veh = NS(get_transform=lambda: tr, is_at_traffic_light=lambda: red,
             get_traffic_light_state=lambda: __import__('bc').carla.TrafficLightState.Red)
    e = NS(route_waypoints=[NS(transform=NS(location=Loc(x=float(k), y=0.0))) for k in range(40)], current_waypoint_index=0,
           vehicle=veh, front_obstacle=front, velocity=v, timesteps=int(round(t / 0.05)),
           infra=NS(warning_now=lambda: warn) if warn is not None else None)
    e.now = lambda: e.timesteps * 0.05
    return e


E = env(np.inf, 15.0, 1.0)
assert expert_action(E)[1] > 0  # clear road: throttle
# a bike crossing 6 m ahead: first 0.3 s only ease off (no brake at 15 km/h toward a 7.5 km/h target... light brake)
E.front_obstacle = 6.0
first = expert_action(E)[1]
assert -0.7 < first < 0, first
# ... and it is still there 0.3 s later: follow / stop behind it
E.timesteps += 6
assert expert_action(E)[1] < first
# it leaves: back to throttle, the timer resets
E.front_obstacle, E.timesteps = np.inf, E.timesteps + 1
assert expert_action(E)[1] > 0 and E.blocked_since is None
# too close: full brake at once
E.front_obstacle = 2.0
assert expert_action(E)[1] == -1.0
# standstill behind something: hold, not full brake
E2 = env(3.0, 0.0, 5.0); expert_action(E2); E2.timesteps += 10
assert expert_action(E2)[1] == -0.5
# DENM: < 1 s -> full brake; 2 s -> slow to 10 km/h (brakes from 18, throttles from 5)
assert expert_action(env(np.inf, 18.0, 1.0, warn=0.8))[1] == -1.0
assert expert_action(env(np.inf, 18.0, 1.0, warn=2.0))[1] < 0 and expert_action(env(np.inf, 5.0, 1.0, warn=2.0))[1] > 0
# red light (we are already at the stop line): hard from 15 km/h, graded from 6, hold at a standstill
p = expert_action(env(np.inf, 6.0, 1.0, red=True))[1]
assert -1.0 < p < 0 and expert_action(env(np.inf, 0.0, 1.0, red=True))[1] == -0.5
# new episode (sim time restarts) with a stale timer: not treated as lingering
E3 = env(6.0, 15.0, 9.0); expert_action(E3); E3.timesteps = 2
assert expert_action(E3)[1] > -0.7
# horn: a bike lingering 5 m ahead while we crawl -> a short tap (0.3 s), then quiet until 2 s later; clear road -> never
H = env(5.0, 3.0, 4.0); expert_action(H); H.timesteps += 7
assert expert_action(H)[2] == 1.0
H.timesteps += 8
assert expert_action(H)[2] == -1.0
H.timesteps += 40
assert expert_action(H)[2] == 1.0
assert expert_action(env(np.inf, 15.0, 1.0))[2] == -1.0 and len(expert_action(env(np.inf, 15.0, 1.0))) == 3
print('ok')
