'''
Autonomous-driving infrastructure (phase 1): roadside units (RSU) with a pole LiDAR at junctions, a simulated V2X
channel, and the messages the ego receives through it.

  CAM   cooperative awareness: every coop car broadcasts its own position (GPS noise) and velocity
  CPM   collective perception: each RSU broadcasts the objects its LiDAR extracts (position, velocity, kind)
  SPaT  signal phase and timing: traffic lights broadcast their state and the seconds left in it
  DENM  decentralized environmental notification: the edge server at a junction warns the ego of a coming conflict

The top half is pure numpy (tested in test_coop.py); the CARLA side (Infrastructure) is at the bottom.
'''
import math
import random
from collections import deque
import numpy as np

INFRA_K = 4  # nearest V2X objects in the observation
INFRA_DIM = INFRA_K * 4 + 5 + 2 + 1  # objects (fwd, right, v_fwd, v_right) + light (dist, red, yellow, green, t_left)
#                                     + DENM warning (active, time to conflict) + staleness


# ------------------------------------------------------------------------------------------------- V2X channel
class V2XChannel:
    '''Broadcast radio with latency + jitter, random loss and a range limit, delivered in simulated time.'''

    def __init__(self, latency=0.10, jitter=0.03, loss=0.10, range_m=300.0, seed=0):
        self.latency, self.jitter, self.loss, self.range = latency, jitter, loss, range_m
        self.rng = random.Random(seed)
        self.queue = []  # (deliver_at, sender_xy, msg)
        self.stats = deque(maxlen=400)  # (t, delivered?, latency) of recent messages, for the dashboard

    def send(self, t, sender_xy, msg):
        delay = max(0.0, self.latency + self.rng.uniform(-self.jitter, self.jitter))
        self.queue.append((t + delay, sender_xy, msg))

    def receive_all(self, t, receivers):
        '''receivers: {name: (x, y)}. Messages due by time t, per receiver: in range and not lost (independently).'''
        due, keep = [], []
        for item in self.queue:
            (due if item[0] <= t else keep).append(item)
        self.queue = keep
        out = {name: [] for name in receivers}
        for at, (sx, sy), msg in due:
            for name, (rx, ry) in receivers.items():
                if msg.get('to') not in (None, name):  # unicast (DENM) only to its addressee
                    continue
                if math.hypot(sx - rx, sy - ry) > self.range:
                    continue
                ok = self.rng.random() >= self.loss
                if name == 'ego':
                    self.stats.append((t, ok, at - msg['t']))
                if ok:
                    out[name].append(msg)
        return out

    def receive(self, t, receiver_xy):
        return self.receive_all(t, {'ego': receiver_xy})['ego']

    def summary(self, t, window=2.0):
        '''Messages per second, delivered share and mean latency over the last `window` s.'''
        recent = [s for s in self.stats if s[0] > t - window]
        ok = [s for s in recent if s[1]]
        return {'msg_per_s': len(ok) / window, 'loss': 1 - len(ok) / len(recent) if recent else 0.0,
                'latency_ms': 1000 * float(np.mean([s[2] for s in ok])) if ok else 0.0}


def closest_approach(p, v, horizon=3.0):
    '''
    Constant-velocity conflict check. p: other position relative to the ego (m), v: other velocity relative to the
    ego (m/s), both [N, 2]. Returns (t, d): time of closest approach in [0, horizon] s and the distance then (m).
    '''
    p, v = np.atleast_2d(np.asarray(p, float)), np.atleast_2d(np.asarray(v, float))
    vv = (v * v).sum(1)
    t = np.where(vv > 1e-6, -(p * v).sum(1) / np.maximum(vv, 1e-6), 0.0)
    t = np.clip(t, 0.0, horizon)
    return t, np.linalg.norm(p + v * t[:, None], axis=1)


def dedupe(objects, radius=1.5):
    '''One entry per physical object: the same bike reported by an RSU and a CAM (or two RSUs) is merged, freshest wins.'''
    out = []
    for o in sorted(objects, key=lambda o: -o['t']):
        if all(math.hypot(o['x'] - q['x'], o['y'] - q['y']) > radius for q in out):
            out.append(o)
    return out


def edge_conflicts(ego, others, horizon=3.0, radius=1.5):
    '''
    Edge server check: ego = (x, y, vx, vy, yaw_deg) from its CAM, others = [{'x','y','vx','vy',...}].
    Conflicts ahead of the ego (meeting point in front of it) within `horizon` s and `radius` m, soonest first.
    '''
    if not others:
        return []
    ex, ey, evx, evy, yaw = ego
    p = np.array([(o['x'] - ex, o['y'] - ey) for o in others])
    v = np.array([(o['vx'] - evx, o['vy'] - evy) for o in others])
    t, d = closest_approach(p, v, horizon)
    meet = p + v * t[:, None] + np.array([evx, evy]) * t[:, None]  # where the other is then, relative to the ego now
    fwd = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    moving = np.array([math.hypot(o['vx'], o['vy']) > 1.0 for o in others])  # parked things are the ego's corridor's job;
    hits = [(float(t[i]), float(d[i]), others[i]) for i in range(len(others))  # with GPS noise they only raise false alarms
            if moving[i] and d[i] < radius and 0 < t[i] < horizon and meet[i] @ fwd > 0]
    return sorted(hits, key=lambda h: h[0])


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
        self.held = set()  # cells kept out of the background: a bike waiting at the kerb must stay an object
        self.born, self.moved, self.t0, self.warmup = {}, set(), None, 3.0

    def __call__(self, pts, ground_z, t):
        '''pts: [N, 3] world points. Returns [{'id', 'x', 'y', 'vx', 'vy', 'kind'}].'''
        self.t0 = t if self.t0 is None else self.t0
        pts = np.asarray(pts, float).reshape(-1, 3)
        pts = pts[(pts[:, 2] > ground_z + self.z_min) & (pts[:, 2] < ground_z + self.z_max)]
        cells = {}
        for c, p in zip(map(tuple, np.floor(pts[:, :2] / self.cell).astype(int)), pts):
            cells.setdefault(c, []).append(p)
        for c in set(self.background) | set(cells):  # exponential average of "this cell is occupied"
            occupied = c in cells and c not in self.held
            self.background[c] = self.background.get(c, 0.0) * (1 - self.rate) + self.rate * occupied
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
            objects.append({'x': float(p[:, 0].mean()), 'y': float(p[:, 1].mean()), 'kind': kind, 'cells': group})
        out = self._track(objects, t)
        # hold only what arrived after the start-up (or ever moved): clutter seen from the first frame (poles, bins,
        # trees) must still turn into background, a truck parking or a bike stopping later must not
        self.held = set()
        for o in out:
            if math.hypot(o['vx'], o['vy']) > 0.5:
                self.moved.add(o['id'])
            if self.born[o['id']] > self.t0 + self.warmup or o['id'] in self.moved:
                self.held.update((gx + dx, gy + dy) for gx, gy in o['cells'] for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        for o in out:
            del o['cells']
        return out

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
                self.born[best] = t
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


def encode_infra_obs(ego_xy, ego_yaw_deg, ego_v, objects, now, light=None, max_age=1.0, warning=None):
    '''
    The ego's V2X view as INFRA_DIM numbers. objects: [{'x','y','vx','vy','t'}] in world, extrapolated to `now`
    with their own velocity; light: (distance to stop line m, state, seconds left) or None;
    warning: seconds to the conflict announced by a DENM, or None.
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
    feats += [0.0, 1.0] if warning is None else [1.0, min(warning, 3.0) / 3.0]
    feats.append(min(np.mean(ages) if ages else max_age, max_age) / max_age)  # how stale the V2X picture is
    return np.array(feats, np.float32)


# ------------------------------------------------------------------------------------------------- CARLA side
class Infrastructure:
    '''RSUs at the junctions, CAM from coop cars, SPaT from lights, all through one V2X channel to the ego.'''

    def __init__(self, world, rsu_height=6.0, max_rsus=8, gps_sigma=1.5, seed=0, cam_cars=1.0, cam_bikes=0.0, **channel):
        '''cam_cars / cam_bikes: share of cars / two-wheelers fitted with V2X (Vietnamese motorbikes: practically none).'''
        from simulation.connection import carla
        from simulation.sensors import LidarSensor
        self.world, self.map, self.carla = world, world.get_map(), carla
        self.channel = V2XChannel(seed=seed, **channel)
        self.gps_sigma, self.rng = gps_sigma, random.Random(seed)
        self.cam_cars, self.cam_bikes, self.connected = cam_cars, cam_bikes, {}  # vehicle id -> fitted with V2X?
        self.enabled = True
        junctions = {}
        for wp in self.map.generate_waypoints(4.0):
            if wp.is_junction:
                j = wp.get_junction()
                junctions[j.id] = j.bounding_box.location
        self.rsus = []
        for jid, loc in list(junctions.items())[:max_rsus]:
            ground = self.map.get_waypoint(loc).transform.location.z
            # infrastructure-grade LiDAR: 64 channels all aimed at the street (0 .. -40 deg; the blind disc under a 6 m pole
            # shrinks from ~13 m at -25 deg to ~7 m), 200k pts/s: a motorbike 20 m away gets ~10 points instead of 1-2
            sensor = LidarSensor(None, carla.Transform(carla.Location(loc.x, loc.y, ground + rsu_height)), world=world,
                                 lower_fov=-40.0, upper_fov=0.0, channels=64, points_per_second=200000)
            self.rsus.append({'junction': jid, 'xy': (loc.x, loc.y), 'ground': ground, 'lidar': sensor,
                              'extract': ObjectExtractor(), 'objects': []})
        self.lights = []  # (light, stop-line waypoint)
        from simulation.vn_traffic import _stop_waypoint
        for tl in world.get_actors().filter('traffic.traffic_light'):
            self.lights.append((tl, _stop_waypoint(self.map, tl)))
        self.known = {}  # (sender, object id) -> object, the ego's V2X world model
        self.light_ahead = None
        self.warning = None  # latest DENM for the ego
        self.edge_cams = {}  # rsu name -> {vehicle id: CAM object}, what each edge server heard
        self.t = 0.0
        self.ticks = 0

    def sensors(self):
        return [r['lidar'].sensor for r in self.rsus]

    def reset(self):
        self.known, self.light_ahead, self.warning = {}, None, None

    def _fitted(self, v):
        if v.id not in self.connected:
            from simulation.vn_traffic import MOTORBIKES, BICYCLES
            share = self.cam_bikes if v.type_id in MOTORBIKES + BICYCLES else self.cam_cars
            self.connected[v.id] = self.rng.random() < share
        return self.connected[v.id]

    def warning_now(self):
        '''Seconds to the conflict of the latest DENM, if still ahead and fresh, else None.'''
        if not self.warning:
            return None
        left = self.warning['ttc'] - (self.t - self.warning['t'])
        return left if left > 0 and self.t - self.warning['t'] < 1.0 else None

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
            for v in self.world.get_actors().filter('vehicle.*'):  # CAM from every V2X-fitted vehicle, the ego included
                if not (v.id == env.vehicle.id or self._fitted(v)):
                    continue
                loc, vel, yaw = v.get_location(), v.get_velocity(), v.get_transform().rotation.yaw
                x, y = loc.x + self.rng.gauss(0, self.gps_sigma), loc.y + self.rng.gauss(0, self.gps_sigma)
                self.channel.send(self.t, (loc.x, loc.y), {'type': 'CAM', 'from': f'car{v.id}', 't': self.t, 'yaw': yaw,
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
        edges = [r for r in self.rsus if math.hypot(r['xy'][0] - ego.x, r['xy'][1] - ego.y) < 80]  # only these can matter
        inbox = self.channel.receive_all(self.t, dict([('ego', (ego.x, ego.y))] + [(f"rsu{r['junction']}", r['xy']) for r in edges]))
        for r in edges:
            self._edge(r, inbox[f"rsu{r['junction']}"], env.vehicle.id)
        spats = {}
        for m in inbox['ego']:
            if m['type'] == 'SPaT':
                spats[m['light']] = m
            elif m['type'] == 'DENM':
                self.warning = m
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

    def _edge(self, r, msgs, ego_id):
        '''Edge server of one RSU: fuse its own LiDAR objects with the CAMs it heard, warn the ego of a conflict.'''
        name = f"rsu{r['junction']}"
        cams = self.edge_cams.setdefault(name, {})
        for m in msgs:
            if m['type'] == 'CAM':
                o = m['objects'][0]
                cams[o['id']] = dict(o, t=m['t'], yaw=m['yaw'])
        cams = {k: o for k, o in cams.items() if self.t - o['t'] < 0.5}
        self.edge_cams[name] = cams
        me = cams.get(ego_id)
        if me is None or self.ticks % 2:
            return
        others = [dict(o, t=self.t) for o in r['objects'] if math.hypot(o['x'] - me['x'], o['y'] - me['y']) > 3.0]  # not the ego itself
        others += [o for k, o in cams.items() if k != ego_id]
        hits = edge_conflicts((me['x'], me['y'], me['vx'], me['vy'], me['yaw']), dedupe(others))
        if hits:
            ttc, dist, o = hits[0]
            self.channel.send(self.t, r['xy'], {'type': 'DENM', 'from': name, 'to': 'ego', 't': self.t, 'ttc': ttc,
                                                'kind': o.get('kind', '?'), 'x': o['x'], 'y': o['y']})

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
        return encode_infra_obs((t.location.x, t.location.y), t.rotation.yaw, (v.x, v.y), dedupe(list(self.known.values())),
                                self.t, light, warning=self.warning_now())

    def destroy(self, client):
        client.apply_batch_sync([self.carla.command.DestroyActor(s) for s in self.sensors()], True)
        self.rsus = []
