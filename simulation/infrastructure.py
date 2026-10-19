'''
Autonomous-driving infrastructure (phase 1): roadside units (RSU) with a pole LiDAR at junctions, a simulated V2X
channel, and the messages the ego receives through it.

  CAM   cooperative awareness: every coop car broadcasts its own position (GPS noise) and velocity
  CPM   collective perception: each RSU broadcasts the objects its LiDAR extracts (position, velocity, kind)
  SPaT  signal phase and timing: traffic lights broadcast their state and the seconds left in it

The top half is pure numpy (tested in test_coop.py); the CARLA side (Infrastructure) is at the bottom.
'''
import math
import random
from collections import deque
import numpy as np

INFRA_K = 4  # nearest V2X objects in the observation
INFRA_DIM = INFRA_K * 4 + 5 + 1  # objects (fwd, right, v_fwd, v_right) + light ahead (dist, red, yellow, green, t_left) + age


# ------------------------------------------------------------------------------------------------- V2X channel
class V2XChannel:
    '''Broadcast radio with latency + jitter, random loss and a range limit, delivered in simulated time.'''

    def __init__(self, latency=0.10, jitter=0.03, loss=0.10, range_m=300.0, seed=0):
        self.latency, self.jitter, self.loss, self.range = latency, jitter, loss, range_m
        self.rng = random.Random(seed)
        self.queue = []  # (deliver_at, sender_xy, msg)
        self.stats = deque(maxlen=400)  # (t, delivered?, latency) of recent messages, for the dashboard

    def send(self, t, sender_xy, msg):
        if self.rng.random() < self.loss:
            self.stats.append((t, False, 0.0))
            return
        delay = max(0.0, self.latency + self.rng.uniform(-self.jitter, self.jitter))
        self.queue.append((t + delay, sender_xy, msg))

    def receive(self, t, receiver_xy):
        '''Messages due by time t from senders within range; out-of-range ones count as lost.'''
        due, keep = [], []
        for item in self.queue:
            (due if item[0] <= t else keep).append(item)
        self.queue = keep
        out = []
        for at, (sx, sy), msg in due:
            ok = math.hypot(sx - receiver_xy[0], sy - receiver_xy[1]) <= self.range
            self.stats.append((t, ok, at - msg['t']))
            if ok:
                out.append(msg)
        return out

    def summary(self, t, window=2.0):
        '''Messages per second, delivered share and mean latency over the last `window` s.'''
        recent = [s for s in self.stats if s[0] > t - window]
        ok = [s for s in recent if s[1]]
        return {'msg_per_s': len(ok) / window, 'loss': 1 - len(ok) / len(recent) if recent else 0.0,
                'latency_ms': 1000 * float(np.mean([s[2] for s in ok])) if ok else 0.0}


# ------------------------------------------------------------------------------------------------- RSU perception
class ObjectExtractor:
    '''
    Pole LiDAR -> objects: drop the ground, learn the static background (walls, poles, trees, parked cars) as
    long-term occupied grid cells, cluster the rest on a grid, track clusters between frames for velocity.
    '''

    def __init__(self, cell=0.8, z_min=0.3, z_max=2.5, background_rate=0.02, max_extent=7.0):
        self.cell, self.z_min, self.z_max, self.rate, self.max_extent = cell, z_min, z_max, background_rate, max_extent
        self.background = {}  # cell -> long-term occupancy in [0, 1]
        self.tracks = {}  # id -> (x, y, vx, vy, t)
        self.next_id = 0

    def __call__(self, pts, ground_z, t):
        '''pts: [N, 3] world points. Returns [{'id', 'x', 'y', 'vx', 'vy', 'kind'}].'''
        pts = np.asarray(pts, float).reshape(-1, 3)
        pts = pts[(pts[:, 2] > ground_z + self.z_min) & (pts[:, 2] < ground_z + self.z_max)]
        cells = {}
        for c, p in zip(map(tuple, np.floor(pts[:, :2] / self.cell).astype(int)), pts):
            cells.setdefault(c, []).append(p)
        for c in set(self.background) | set(cells):  # exponential average of "this cell is occupied"
            self.background[c] = self.background.get(c, 0.0) * (1 - self.rate) + self.rate * (c in cells)
        moving = {c for c in cells if self.background.get(c, 0.0) < 0.5}
        objects, seen = [], set()
        for c in moving:  # flood fill 8-connected cells into clusters
            if c in seen:
                continue
            stack, group = [c], []
            seen.add(c)
            while stack:
                cx, cy = stack.pop()
                group.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (cx + dx, cy + dy)
                        if n in moving and n not in seen:
                            seen.add(n)
                            stack.append(n)
            p = np.concatenate([cells[g] for g in group])
            extent = np.ptp(p[:, :2], axis=0)
            if len(p) < 4 or extent.max() > self.max_extent:  # noise, or a building edge the background missed
                continue
            kind = 'car' if extent.max() > 2.8 else 'two-wheeler' if extent.max() > 1.2 else 'pedestrian'
            objects.append({'x': float(p[:, 0].mean()), 'y': float(p[:, 1].mean()), 'kind': kind})
        return self._track(objects, t)

    def _track(self, objects, t, gate=2.5):
        '''Nearest-neighbour association with the previous frame; velocity smoothed over frames.'''
        out, used = [], set()
        for o in objects:
            best, bd = None, gate
            for tid, (x, y, vx, vy, tt) in self.tracks.items():
                d = math.hypot(o['x'] - (x + vx * (t - tt)), o['y'] - (y + vy * (t - tt)))
                if tid not in used and d < bd:
                    best, bd = tid, d
            if best is None:
                best, vx, vy = self.next_id, 0.0, 0.0
                self.next_id += 1
            else:
                x, y, pvx, pvy, tt = self.tracks[best]
                dt = max(t - tt, 1e-3)
                vx, vy = 0.6 * pvx + 0.4 * (o['x'] - x) / dt, 0.6 * pvy + 0.4 * (o['y'] - y) / dt
            used.add(best)
            self.tracks[best] = (o['x'], o['y'], vx, vy, t)
            out.append(dict(o, id=best, vx=vx, vy=vy))
        self.tracks = {k: v for k, v in self.tracks.items() if k in used or t - v[4] < 1.0}
        return out


# ------------------------------------------------------------------------------------------------- ego observation
def light_phase_left(state, elapsed, green, yellow, red):
    '''Seconds left in the current phase of a light.'''
    return max(0.0, {'Green': green, 'Yellow': yellow, 'Red': red}.get(state, 0.0) - elapsed)


def group_red_left(me, active, active_left, phases, clearance=0.0):
    '''
    Seconds until light `me` of a CARLA light group turns green. Groups cycle one light at a time
    (green -> yellow -> next light); `active` is the index of the current green/yellow light with `active_left`
    seconds of green+yellow left, phases[i] = (green, yellow) of light i, clearance: all-red time between turns.
    '''
    n, left, i = len(phases), active_left + clearance, (active + 1) % len(phases)
    while i != me and n:
        left += sum(phases[i]) + clearance
        i = (i + 1) % n
    return left


def encode_infra_obs(ego_xy, ego_yaw_deg, ego_v, objects, now, light=None, max_age=1.0):
    '''
    The ego's V2X view as INFRA_DIM numbers. objects: [{'x','y','vx','vy','t'}] in world, extrapolated to `now`
    with their own velocity; light: (distance to stop line m, state, seconds left) or None.
    '''
    yaw = math.radians(ego_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rows, ages = [], []
    for o in objects:
        age = now - o['t']
        if age > max_age:
            continue
        dx, dy = o['x'] + o['vx'] * age - ego_xy[0], o['y'] + o['vy'] * age - ego_xy[1]
        f, r = dx * c + dy * s, -dx * s + dy * c
        vf, vr = (o['vx'] - ego_v[0]) * c + (o['vy'] - ego_v[1]) * s, -(o['vx'] - ego_v[0]) * s + (o['vy'] - ego_v[1]) * c
        rows.append((math.hypot(f, r), [f / 30, r / 30, vf / 10, vr / 10]))
        ages.append(age)
    rows.sort(key=lambda rw: rw[0])
    feats = [v for _, row in rows[:INFRA_K] for v in row] + [0.0, 0.0, 0.0, 0.0] * (INFRA_K - min(len(rows), INFRA_K))
    if light is None:
        feats += [1.0, 0.0, 0.0, 0.0, 0.0]  # no light ahead: "far away", no colour
    else:
        dist, state, left = light
        feats += [min(dist, 50) / 50, state == 'Red', state == 'Yellow', state == 'Green', min(left, 30) / 30]
    feats.append(min(np.mean(ages) if ages else max_age, max_age) / max_age)  # how stale the V2X picture is
    return np.array(feats, np.float32)


# ------------------------------------------------------------------------------------------------- CARLA side
class Infrastructure:
    '''RSUs at the junctions, CAM from coop cars, SPaT from lights, all through one V2X channel to the ego.'''

    def __init__(self, world, rsu_height=6.0, max_rsus=8, gps_sigma=1.5, seed=0, **channel):
        from simulation.connection import carla
        from simulation.sensors import LidarSensor
        self.world, self.map, self.carla = world, world.get_map(), carla
        self.channel = V2XChannel(seed=seed, **channel)
        self.gps_sigma, self.rng = gps_sigma, random.Random(seed)
        self.enabled = True
        junctions = {}
        for wp in self.map.generate_waypoints(4.0):
            if wp.is_junction:
                j = wp.get_junction()
                junctions[j.id] = j.bounding_box.location
        self.rsus = []
        for jid, loc in list(junctions.items())[:max_rsus]:
            ground = self.map.get_waypoint(loc).transform.location.z
            # -40 deg: from a 6 m pole the blind disc under it shrinks from ~13 m (-25 deg) to ~7 m
            sensor = LidarSensor(None, carla.Transform(carla.Location(loc.x, loc.y, ground + rsu_height)), world=world, lower_fov=-40.0)
            self.rsus.append({'junction': jid, 'xy': (loc.x, loc.y), 'ground': ground, 'lidar': sensor,
                              'extract': ObjectExtractor(), 'objects': []})
        self.lights = []  # (light, stop-line waypoint)
        from simulation.vn_traffic import _stop_waypoint
        for tl in world.get_actors().filter('traffic.traffic_light'):
            self.lights.append((tl, _stop_waypoint(self.map, tl)))
        self.known = {}  # (sender, object id) -> object, the ego's V2X world model
        self.light_ahead = None
        self.t = 0.0
        self.ticks = 0

    def sensors(self):
        return [r['lidar'].sensor for r in self.rsus]

    def reset(self):
        self.known, self.light_ahead = {}, None

    def update(self, env):
        '''Once per tick (0.05 s): RSU perception + CPM at 10 Hz, CAM at 10 Hz, SPaT at 2 Hz, ego receives.'''
        self.t += 0.05
        self.ticks += 1
        if not self.enabled:
            return
        if self.ticks % 2 == 0:
            for r in self.rsus:
                if r['lidar'].latest is None:
                    continue
                from simulation.coop import to_ego
                pts = to_ego(r['lidar'].latest[0], r['lidar'].latest[1], np.eye(4))
                r['objects'] = r['extract'](pts, r['ground'], self.t)
                self.channel.send(self.t, r['xy'], {'type': 'CPM', 'from': f"rsu{r['junction']}", 't': self.t,
                                                    'objects': r['objects']})
            for i, l in enumerate(env.lidar_objs[1:]):  # coop cars: CAM with GPS noise
                v = l.parent
                if v is None or not v.is_alive:
                    continue
                loc, vel = v.get_location(), v.get_velocity()
                x, y = loc.x + self.rng.gauss(0, self.gps_sigma), loc.y + self.rng.gauss(0, self.gps_sigma)
                self.channel.send(self.t, (loc.x, loc.y), {'type': 'CAM', 'from': f'car{v.id}', 't': self.t,
                                                          'objects': [{'id': v.id, 'x': x, 'y': y, 'vx': vel.x, 'vy': vel.y, 'kind': 'car'}]})
        if self.ticks % 10 == 0:
            for tl, stop in self.lights:
                if stop is None:
                    continue
                st = str(tl.get_state())
                left = self._time_left(tl, st)
                loc = stop.transform.location
                self.channel.send(self.t, (loc.x, loc.y), {'type': 'SPaT', 'from': f'tl{tl.id}', 't': self.t,
                                                          'light': tl.id, 'state': st, 'left': left, 'stop': (loc.x, loc.y)})
        ego = env.vehicle.get_location()
        spats = {}
        for m in self.channel.receive(self.t, (ego.x, ego.y)):
            if m['type'] == 'SPaT':
                spats[m['light']] = m
            else:
                for o in m['objects']:
                    if m['type'] == 'CAM' and o['id'] == env.vehicle.id:
                        continue
                    self.known[(m['from'], o['id'])] = dict(o, t=m['t'], src=m['type'])
        self.known = {k: o for k, o in self.known.items() if self.t - o['t'] < 1.0}
        if spats:
            self.light_ahead = self._light_on_route(env, spats) or self.light_ahead
        if self.light_ahead and self.t - self.light_ahead[3] > 1.5:
            self.light_ahead = None

    def _time_left(self, tl, state):
        '''Seconds left in the light's state; red waits for the other lights of its group to go round.'''
        if state != 'Red':
            return light_phase_left(state, tl.get_elapsed_time(), tl.get_green_time(), tl.get_yellow_time(), tl.get_red_time())
        group = list(tl.get_group_traffic_lights())
        ids = [g.id for g in group]
        active = next((k for k, g in enumerate(group) if str(g.get_state()) != 'Red'), None)
        if tl.id not in ids or active is None:
            return light_phase_left(state, tl.get_elapsed_time(), tl.get_green_time(), tl.get_yellow_time(), tl.get_red_time())
        a = group[active]
        a_left = light_phase_left(str(a.get_state()), a.get_elapsed_time(), a.get_green_time(), a.get_yellow_time(), a.get_red_time())
        if str(a.get_state()) == 'Green':
            a_left += a.get_yellow_time()
        return group_red_left(ids.index(tl.id), active, a_left, [(g.get_green_time(), g.get_yellow_time()) for g in group],
                              clearance=tl.get_red_time())  # all lights red for red_time between two turns (measured)

    def _light_on_route(self, env, spats, ahead_m=60):
        '''The first light whose stop line lies on the ego route within `ahead_m`: (distance, state, left, t).'''
        i = env.current_waypoint_index
        route = env.route_waypoints[i:i + ahead_m]
        for k, wp in enumerate(route):
            p = wp.transform.location
            for m in spats.values():
                if math.hypot(m['stop'][0] - p.x, m['stop'][1] - p.y) < 2.0:
                    return (float(k), m['state'], m['left'], m['t'])
        return None

    def obs(self, env):
        t = env.vehicle.get_transform()
        v = env.vehicle.get_velocity()
        light = self.light_ahead[:3] if self.light_ahead else None
        if light:  # the distance was measured when the SPaT arrived: correct it by what the ego drove since
            light = (max(0.0, light[0] - env.velocity / 3.6 * (self.t - self.light_ahead[3])), light[1], max(0.0, light[2] - (self.t - self.light_ahead[3])))
        return encode_infra_obs((t.location.x, t.location.y), t.rotation.yaw, (v.x, v.y), list(self.known.values()), self.t, light)

    def destroy(self, client):
        client.apply_batch_sync([self.carla.command.DestroyActor(s) for s in self.sensors()], True)
        self.rsus = []
