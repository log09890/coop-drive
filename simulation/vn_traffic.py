'''
Chaotic Vietnamese-style urban traffic for CARLA 0.9.10 (sync mode, Traffic Manager on port 8000):
mostly motorbikes, tailgating, mixed speeds, some red-light running and sign ignoring, motorbikes weaving
between lanes, and pedestrians crossing anywhere.
'''
import math
import random
import numpy as np
from simulation.connection import carla

MOTORBIKES = ('vehicle.yamaha.yzf', 'vehicle.kawasaki.ninja', 'vehicle.harley-davidson.low_rider')
BICYCLES = ('vehicle.bh.crossbike', 'vehicle.gazelle.omafiets', 'vehicle.diamondback.century')
CARS = ('vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.nissan.micra', 'vehicle.mini.cooperst', 'vehicle.seat.leon')

# per kind: (share of red lights ignored %, signs ignored %, gap to leader m, speed vs limit %: negative = faster)
STYLE = {'moto': (30, 60, (1.5, 3.0), (-40, 10)), 'bike': (40, 70, (1.5, 2.5), (40, 70)), 'car': (10, 30, (2.0, 4.0), (-15, 20))}


class VNTraffic:

    def __init__(self, client, world, n_motorbikes=80, n_bicycles=8, n_cars=12, n_walkers=50, avoid=None, seed=0):
        self.client, self.world, self.rng = client, world, random.Random(seed)
        self.map = world.get_map()  # get_map() re-downloads the OpenDRIVE map on every call
        self.tm = client.get_trafficmanager(8000)
        self.tm.set_hybrid_physics_mode(True)  # no physics for TM vehicles far from the ego ('hero'): fewer pile-ups, faster
        self.tm.set_hybrid_physics_radius(70.0)
        bl = world.get_blueprint_library()
        # lane points every 6 m (CODD-style): many more slots than spawn points, motorbikes need lots of them
        self.slots = [w.transform for w in self.map.generate_waypoints(6.0)]
        slots = [s for s in self.slots if avoid is None or s.location.distance(avoid) > 15]  # keep the ego spawn free
        self.rng.shuffle(slots)
        kinds = ['moto'] * n_motorbikes + ['bike'] * n_bicycles + ['car'] * n_cars
        self.vehicles, self.motorbikes = [], []
        for kind, tr in zip(kinds, slots):
            self._spawn(kind, tr)

        world.set_pedestrians_cross_factor(0.6)  # 60 % of walkers cross the road anywhere, not only at crossings
        walker_bps = bl.filter('walker.pedestrian.*')
        self.walkers, self.controllers = [], []
        for _ in range(n_walkers):
            loc = world.get_random_location_from_navigation()
            w = world.try_spawn_actor(self.rng.choice(walker_bps), carla.Transform(loc)) if loc else None
            if w is not None:
                self.walkers.append(w)
                self.controllers.append(world.spawn_actor(bl.find('controller.ai.walker'), carla.Transform(), w))
        world.tick()  # walker controllers can only start after a tick
        for c in self.controllers:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(self.rng.uniform(1.0, 2.2))  # some walk, some hurry
        self.lights = [(tl, _stop_waypoint(self.map, tl)) for tl in world.get_actors().filter('traffic.traffic_light')]
        for tl, _ in self.lights:  # default cycle keeps each light red ~39 s; ~16 s keeps the traffic moving
            tl.set_green_time(4.0)
            tl.set_yellow_time(1.5)
        self.swarm = {}  # vehicle id -> (light id, cells) for vehicles packed in front of a red light
        self.ticks = self.rescued = 0
        self.stall = {}  # vehicle id -> ticks standing still without a reason
        self.chaos = 5.0  # mean seconds between random surprise events near the ego, 0 = off
        self.next_event = int(self.chaos / DT)
        self.log = []  # event names for the dashboard

    def _spawn(self, kind, tr):
        tr = carla.Transform(tr.location + carla.Location(z=0.3), tr.rotation)
        bp = self.world.get_blueprint_library().find(self.rng.choice({'moto': MOTORBIKES, 'bike': BICYCLES, 'car': CARS}[kind]))
        v = self.world.try_spawn_actor(bp, tr)
        if v is None:
            return None
        lights, signs, gap, speed = STYLE[kind]
        v.set_autopilot(True, 8000)
        self.tm.ignore_lights_percentage(v, lights)
        self.tm.ignore_signs_percentage(v, signs)
        self.tm.distance_to_leading_vehicle(v, self.rng.uniform(*gap))
        self.tm.vehicle_percentage_speed_difference(v, self.rng.uniform(*speed))
        self.tm.auto_lane_change(v, True)
        self.vehicles.append(v)
        if kind == 'moto':
            self.motorbikes.append(v)
        return v

    def _near_ego_slots(self, env, lo=25, hi=70):
        ego = env.vehicle.get_location()
        return [s for s in self.slots if lo < s.location.distance(ego) < hi]

    def set_motorbikes(self, env, n):
        '''/density: add motorbikes around the ego, or remove the farthest ones.'''
        if n > len(self.motorbikes):
            near = self._near_ego_slots(env, 15, 80)
            self.rng.shuffle(near)
            for tr in near[:n - len(self.motorbikes)]:
                self._spawn('moto', tr)
        else:
            ego = env.vehicle.get_location()
            far = sorted((v for v in self.motorbikes if v.id not in self.swarm), key=lambda v: -v.get_location().distance(ego))
            gone = far[:len(self.motorbikes) - n]
            self.client.apply_batch_sync([carla.command.DestroyActor(v) for v in gone], True)
            ids = {v.id for v in gone}
            self.vehicles = [v for v in self.vehicles if v.id not in ids]
            self.motorbikes = [v for v in self.motorbikes if v.id not in ids]
        return len(self.motorbikes)

    def _recycle(self, env):
        '''Keep the traffic dense where it matters: motorbikes far from the ego reappear 25-70 m from it.'''
        ego = env.vehicle.get_location()
        near = self._near_ego_slots(env)
        occupied = [a.get_location() for a in self.world.get_actors().filter('vehicle.*')]
        for v in self.motorbikes:
            if near and v.id not in self.swarm and v.get_location().distance(ego) > 90:
                tr = self.rng.choice(near)
                if all(tr.location.distance(o) > 4 for o in occupied):  # never drop it onto another vehicle
                    v.set_transform(carla.Transform(tr.location + carla.Location(z=0.3), tr.rotation))
                    occupied.append(tr.location)

    def _rescue(self, env):
        """Safety net: a vehicle knocked over or launched off the road is put back on a free lane spot."""
        occupied = [o for _, o in env.vehicle_locs]
        riding = {r.v.id for r in env.riders if r.v}
        far = self._near_ego_slots(env, 40, 90) or self.slots
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.id == env.vehicle.id:
                continue
            t = a.get_transform()
            wp = self.map.get_waypoint(t.location)
            vel = a.get_velocity()
            waiting = a.id in self.swarm or a.id in riding or (
                a.is_at_traffic_light() and a.get_traffic_light_state() != carla.TrafficLightState.Green)
            self.stall[a.id] = self.stall.get(a.id, 0) + 10 if (vel.x ** 2 + vel.y ** 2) < 0.25 and not waiting else 0
            stuck = self.stall[a.id] > 400  # 20 s still for no reason (TM dead end, wedged between bikes)
            if not stuck and abs(t.rotation.pitch) < 35 and abs(t.rotation.roll) < 35 and t.location.z - wp.transform.location.z < 1.5:
                continue
            self.stall[a.id] = 0
            for tr in self.rng.sample(far, min(20, len(far))):
                if all(tr.location.distance(o) > 5 for o in occupied):
                    a.set_transform(carla.Transform(tr.location + carla.Location(z=0.3), tr.rotation))
                    a.set_target_velocity(carla.Vector3D())
                    occupied.append(tr.location)
                    self.rescued += 1
                    break

    def update(self, env):
        '''Call once per world tick: motorbikes swerve between lanes, and every `chaos` s a surprise near the ego.'''
        self.ticks += 1
        if self.chaos and self.ticks >= self.next_event:
            self.next_event = self.ticks + int(self.chaos * self.rng.uniform(0.5, 1.5) / DT)  # irregular: harder to anticipate
            name = self.rng.choice(list(EVENTS))
            if EVENTS[name](env):
                self.log.append(name)
        if self.ticks % 40 == 20:
            self._recycle(env)
        if self.ticks % 10 == 5:
            self._rescue(env)
        if self.ticks % 5 == 0:
            self._swarm(env)
        if self.ticks % 40 == 0 and self.motorbikes:
            for v in self.rng.sample(self.motorbikes, max(1, len(self.motorbikes) // 5)):
                self.tm.force_lane_change(v, self.rng.random() < 0.5)

    def _swarm(self, env):
        '''At red lights nobody queues: vehicles approaching the stop line are packed into free cells of a grid
        that starts past the line and spreads over the whole road, the oncoming lane included; cars take 2x2 cells.'''
        others = [v for v in self.vehicles if v.id not in self.swarm] + [env.vehicle]
        ego = env.vehicle.get_location()
        for tl, stop in self.lights:
            if stop is None or tl.get_state() == carla.TrafficLightState.Green or stop.transform.location.distance(ego) > 80:
                continue  # far lights keep plain TM queues: scripted swarms cost a Python update per vehicle and tick
            f, r, o = stop.transform.get_forward_vector(), stop.transform.get_right_vector(), stop.transform.location
            half = stop.lane_width / 2
            taken = {c for vid, (lid, cells) in self.swarm.items() if lid == tl.id for c in cells}
            blockers = []  # where non-swarm vehicles (and the ego, with a wider margin) stand: (along, lateral, margins)
            for v in others:
                d = v.get_location() - o
                m = (4.0, 2.0) if v.id == env.vehicle.id else (2.5, 1.5)
                blockers.append((d.x * f.x + d.y * f.y, d.x * r.x + d.y * r.y, m))
            for v in self.vehicles:
                if v.id in self.swarm:
                    continue
                d = v.get_location() - o
                along, lat = d.x * f.x + d.y * f.y, d.x * r.x + d.y * r.y
                vf = v.get_transform().get_forward_vector()
                if not (-40 < along < 0 and abs(lat) < stop.lane_width and vf.x * f.x + vf.y * f.y > 0.7):
                    continue
                car = v.type_id not in MOTORBIKES + BICYCLES
                cells = _free_cells(taken, blockers, half, 2 if car else 1, along)
                if cells is None:
                    continue
                taken |= set(cells)
                row, col = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
                goal_along, goal_lat = 1.5 - CELL_D * row, half - 0.6 - CELL_W * col
                self.swarm[v.id] = (tl.id, cells)
                Rider(env, env.map.get_waypoint(v.get_location()), self._swarm_plan(tl, stop, goal_along, goal_lat, v.id), life=120, vehicle=v)

    def _swarm_plan(self, tl, stop, goal_along, goal_lat, vid):
        f, o = stop.transform.get_forward_vector(), stop.transform.location

        def plan(r):
            if tl.get_state() == carla.TrafficLightState.Green:
                self.swarm.pop(vid, None)
                return None
            d = r.v.get_location() - o
            gap = goal_along - (d.x * f.x + d.y * f.y)
            return goal_lat, max(0.0, min(8.0, 3.0 * gap)) if gap > 0.3 else 0.0  # creep up and stop in the cell
        return plan

    def destroy(self):
        for c in self.controllers:
            c.stop()
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in self.controllers + self.walkers + self.vehicles], True)
        self.vehicles, self.motorbikes, self.walkers, self.controllers = [], [], [], []


CELL_W, CELL_D = 1.25, 2.6  # swarm grid: a motorbike per cell with ~0.4 m to spare, a car takes 2 x 2


def _stop_waypoint(world_map, tl):
    '''Lane waypoint at the stop line of a traffic light (0.9.10 has no get_stop_waypoints): from its trigger volume
    move forward until the next step would enter the junction.'''
    try:
        wp = world_map.get_waypoint(tl.get_transform().transform(tl.trigger_volume.location))
    except RuntimeError:
        return None
    for _ in range(40):
        nxt = wp.next(0.5)
        if not nxt or nxt[0].is_junction:
            break
        wp = nxt[0]
    return wp


def _free_cells(taken, blockers, half, size, along_now, rows=12):
    '''First free size x size block of cells, front row first. Cell (row, col): CELL_D deep from 1.5 m past the stop
    line backwards, CELL_W wide from the right lane edge leftwards over the centre line (lane width + 1.5 m).'''
    cols = int((2 * half + 1.5) // CELL_W)
    for row in range(rows):
        if 1.5 - CELL_D * row < along_now - 1:  # never ask a vehicle to back up
            return None
        for col in range(cols - size + 1):
            block = [(row + i, col + j) for i in range(size) for j in range(size)]
            if any(c in taken for c in block):
                continue
            ga, gl = 1.5 - CELL_D * (row + (size - 1) / 2), half - 0.6 - CELL_W * (col + (size - 1) / 2)
            if any(abs(ga - a) < ma and abs(gl - l) < ml for a, l, (ma, ml) in blockers):
                continue
            return block
    return None


# ---------------------------------------------------------------------------------------------
# Reckless riders: scripted motorbikes for what the Traffic Manager will not do on 1-lane roads
# ---------------------------------------------------------------------------------------------

DT = 0.05  # sync tick


def _offset(wp, d):
    '''Point d metres to the right (negative: left) of a waypoint, on the road.'''
    r = wp.transform.get_right_vector()
    return wp.transform.location + carla.Location(x=r.x * d, y=r.y * d)


def _ahead_of_ego(env, actor):
    d, f = actor.get_location() - env.vehicle.get_location(), env.vehicle.get_transform().get_forward_vector()
    return d.x * f.x + d.y * f.y


class Rider:
    '''Motorbike following a reference lane at plan(rider) -> (lateral offset m, speed km/h), then handed to the TM.'''

    def __init__(self, env, wp, plan, life, vehicle=None):
        '''vehicle=None spawns a new motorbike (dies with the episode); else takes over that vehicle from the TM.'''
        self.env, self.wp, self.plan, self.life, self.t, self.v = env, wp, plan, life, 0.0, vehicle
        self.last_control = None
        self.merging = False
        if vehicle is None:
            bp = env.world.get_blueprint_library().find(random.choice(MOTORBIKES))
            tr = carla.Transform(_offset(wp, plan(self)[0]) + carla.Location(z=0.5), wp.transform.rotation)
            self.v = env.world.try_spawn_actor(bp, tr)
            if self.v:
                env.actor_list.append(self.v)
        else:
            vehicle.set_autopilot(False)
        if self.v:
            env.riders.append(self)  # stepped by the env every tick

    def _blocked(self, loc, tr, gap=1.0):
        '''Someone within `gap` m of our front, bumper to bumper (a car's centre is 2.4 m from its rear).'''
        f, e = tr.get_forward_vector(), self.v.bounding_box.extent
        d = self.env.vehicle_xy - (loc.x, loc.y)
        a, lat = d @ (f.x, f.y), d @ (-f.y, f.x)
        ext = self.env.vehicle_ext
        return bool(np.any((a > 0) & (a < e.x + ext[:, 0] + gap) & (np.abs(lat) < e.y + ext[:, 1] + 0.25)
                           & (self.env.vehicle_ids != self.v.id)))

    def __bool__(self):
        return self.v is not None  # spawn failed (spot occupied) -> falsy like a failed try_spawn_actor

    def step(self):
        '''One tick; False when done (handed over to autopilot or gone).'''
        self.t += DT
        if self.v is None or not self.v.is_alive:
            return False
        if self.t > self.life:
            if not self.merging:  # 3 s back to the lane centre first: the TM strands vehicles it finds off the lane
                self.merging, self.life, self.plan = True, self.life + 3.0, lambda r: (0.0, 18.0)
            else:
                self.v.set_autopilot(True, 8000)
                return False
        loc = self.v.get_location()
        for _ in range(20):  # advance the reference waypoint along its lane
            f, d = self.wp.transform.get_forward_vector(), loc - self.wp.transform.location
            nxt = self.wp.next(1.0)
            if not nxt or d.x * f.x + d.y * f.y < 1.0:
                break
            self.wp = nxt[0]
        plan = self.plan(self)
        if plan is None:  # plan is over (e.g. light turned green)
            self.v.set_autopilot(True, 8000)
            return False
        offset, speed = plan
        if self._blocked(loc, self.v.get_transform()):
            speed = 0.0  # never ride into someone: stacked bikes are what launches them into the air
        ahead = self.wp.next(5.0)
        if not ahead:
            self.v.set_autopilot(True, 8000)
            return False
        target, t = _offset(ahead[0], offset), self.v.get_transform()
        fwd = t.get_forward_vector()
        angle = math.atan2(target.y - loc.y, target.x - loc.x) - math.atan2(fwd.y, fwd.x)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        v = math.hypot(self.v.get_velocity().x, self.v.get_velocity().y) * 3.6
        err = speed - v
        if speed == 0 and v < 0.5:  # parked in its swarm cell / waiting: hold the brake, no new command needed
            control = (0.0, 0.0, 1.0)
        else:
            control = (round(max(-1.0, min(1.0, 2.0 * angle)), 2), round(max(0.0, min(1.0, 0.1 * err)), 2) if err > 0 else 0.0,
                       round(max(0.0, min(1.0, -0.15 * err)), 2) if err < -2 else 0.0)
        if control != self.last_control:  # one RPC less per rider and tick when nothing changes
            self.v.apply_control(carla.VehicleControl(steer=control[0], throttle=control[1], brake=control[2]))
            self.last_control = control
        return True


def _ego_lane(env, metres):
    wp = env.map.get_waypoint(env.vehicle.get_location())
    nxt = wp.next(metres) if metres > 0 else wp.previous(-metres)
    return nxt[0] if nxt else None


def cutin(env):
    '''Rides the right edge just ahead, swerves into the ego path, then dawdles.'''
    wp = _ego_lane(env, 12)
    return wp and Rider(env, wp, lambda r: (1.4 * max(0.0, 1 - max(0.0, r.t - 1.0)), 18 if r.t < 2 else 8), life=6)


def encroach(env):
    '''Oncoming motorbike riding over the centre line into the ego lane.'''
    wp = _ego_lane(env, 35)
    lane = wp and wp.get_left_lane()
    return lane and Rider(env, lane, lambda r: (-1.8 if r.t < 4 else 0.0, 30), life=6)


def overtake(env):
    '''Fast motorbike from behind, passes on the oncoming side, cuts back in front of the ego.'''
    wp = _ego_lane(env, -10)

    def plan(r):
        a = _ahead_of_ego(env, r.v) if r.v else -10
        back = max(0.0, min(1.0, (a - 5) / 4))  # 0 until 5 m ahead, fully back in lane at 9 m
        return -3.2 * (1 - back), max(30.0, env.velocity + 15) if back < 1 else max(5.0, env.velocity - 3)
    return wp and Rider(env, wp, plan, life=10)


def suddenstop(env):
    '''Motorbike ahead brakes hard for no visible reason, then rides off.'''
    wp = _ego_lane(env, 14)
    return wp and Rider(env, wp, lambda r: (0.0, max(15.0, env.velocity) if r.t < 1.5 else 0.0 if r.t < 4.5 else 20), life=7)


def jaywalk(env):
    '''Pedestrian sprints across the road close ahead.'''
    wp = _ego_lane(env, 14)
    if not wp:
        return None
    r = wp.transform.get_right_vector()
    w = env.world.try_spawn_actor(env.world.get_blueprint_library().filter('walker.pedestrian.*')[0],
                                  carla.Transform(_offset(wp, 4.5) + carla.Location(z=1.0)))
    if w:
        w.apply_control(carla.WalkerControl(direction=carla.Vector3D(-r.x, -r.y, 0), speed=3.5))
        env.actor_list.append(w)
    return w


def swerve(env):
    '''Motorbike riding on the left, next to/just ahead of the ego, suddenly cuts across to the right sidewalk.'''
    wp = _ego_lane(env, 7)
    if not wp:
        return None
    left, kerb = -(wp.lane_width / 2 + 0.8), wp.lane_width / 2 + 1.8

    def plan(r):
        k = max(0.0, min(1.0, (r.t - 1.0) / 1.2))  # 1 s alongside, then 1.2 s across the ego path
        return left + (kerb - left) * k, max(12.0, env.velocity + 4) if k < 1 else 5.0
    return Rider(env, wp, plan, life=6)


def wrongway(env):
    '''Motorbike riding against the traffic along the right edge of the ego lane.'''
    wp = _ego_lane(env, 40)
    lane = wp and wp.get_left_lane()
    return lane and Rider(env, lane, lambda r: (-(1.5 * lane.lane_width - 0.7), 20), life=9)


def pullout(env):
    '''Motorbike darts out of the right sidewalk into the lane just ahead.'''
    wp = _ego_lane(env, 11)
    if not wp:
        return None
    kerb = wp.lane_width / 2 + 1.8
    return Rider(env, wp, lambda r: (kerb * max(0.0, 1 - max(0.0, r.t - 0.3) / 0.8), 12.0), life=6)


def zigzag(env):
    '''Motorbike ahead weaving left and right across the lane.'''
    wp = _ego_lane(env, 13)
    return wp and Rider(env, wp, lambda r: (1.3 * math.sin(2.2 * r.t), max(12.0, env.velocity)), life=9)


def flock(env):
    '''A pack of motorbikes passes the ego on both sides and closes in ahead of it.'''
    riders = []
    for back, side in ((-8, -2.6), (-10, 1.6), (-12, -3.4), (-14, 1.8), (-16, -2.0)):
        wp = _ego_lane(env, back)
        if not wp:
            continue

        def plan(r, side=side, slot=random.uniform(-0.8, 0.8)):  # each rider merges into its own spot
            a = _ahead_of_ego(env, r.v) if r.v else -10
            merge = max(0.0, min(1.0, (a - 4) / 6))
            return side * (1 - merge) + slot * merge, max(25.0, env.velocity + 12) if merge < 1 else max(8.0, env.velocity)
        r = Rider(env, wp, plan, life=10)
        if r:
            riders.append(r)
    return riders


def surround(env):
    """Motorbikes box the ego in on all sides and hold the formation ~10 s before drifting off ahead."""
    riders = []
    for along, lat in ((7.5, -0.3), (7.0, 1.8), (0.5, -2.3), (0.0, 2.3), (-2.5, -2.3), (-5.5, 0.3), (-5.0, 1.9)):
        wp = _ego_lane(env, along) if along else env.map.get_waypoint(env.vehicle.get_location())
        if not wp:
            continue

        def plan(r, along=along, lat=lat):
            if r.t > 10:  # break up: everybody overtakes and leaves, keeping its side so nobody cuts into the ego
                return lat, max(20.0, env.velocity + 10)
            a = _ahead_of_ego(env, r.v) if r.v else along
            speed = max(0.0, min(25.0, env.velocity + 2.0 * (along - a)))  # keep its place around the ego
            return lat, max(5.0, speed) if along > 3 else speed  # the bikes in front keep drifting on, so the ego can creep after them
        r = Rider(env, wp, plan, life=14)
        if r:
            riders.append(r)
    return riders


EVENTS = {'cutin': cutin, 'encroach': encroach, 'overtake': overtake, 'suddenstop': suddenstop, 'jaywalk': jaywalk,
          'swerve': swerve, 'wrongway': wrongway, 'pullout': pullout, 'zigzag': zigzag, 'flock': flock, 'surround': surround}
