'''
Occlusion benchmark: a motorbike hidden in front of a parked truck darts across the ego lane (vn_traffic.occluded).
Measures how early the ego knows about the bike, by channel, and what happens, for one configuration:

  ego   own LiDAR only                    (share off, infra off)
  v2v   + coop cars' LiDAR                (share on,  infra off)
  v2i   + roadside units, CPM, DENM        (share off, infra on)
  both  everything                        (share on,  infra on)

  python occlusion_eval.py --config both --trials 10       -> results/occlusion_both.json
  python occlusion_eval.py --table                          -> table of every results/occlusion_*.json

The waypoint expert drives: it brakes for what its lane corridor sees (own or shared LiDAR) and for DENM warnings.
'''
import argparse
import glob
import json
import math
import os
import random
import numpy as np

CONFIGS = {'ego': (False, False), 'v2v': (True, False), 'v2i': (False, True), 'both': (True, True)}
DT = 0.05


def lead_time(t_aware, t_go, t_cross):
    '''
    Seconds of warning the ego had before the bike entered its lane, counted from when the bike starts to dart
    (knowing a parked bike for a minute is no more useful than knowing it when it moves). Negative: found out only
    after it was in the lane; None: never knew, or the bike never crossed.
    '''
    if t_aware is None or t_cross is None or t_go is None:
        return None
    return t_cross - max(t_aware, t_go)


def summarize(trials):
    leads = [t['lead'] for t in trials if t['lead'] is not None]
    return {'trials': len(trials),
            'aware_at_dart_%': round(100 * np.mean([t['aware_at_dart'] for t in trials]), 1) if trials else 0,
            'aware_before_crossing_%': round(100 * np.mean([t['lead'] is not None and t['lead'] > 0 for t in trials]), 1) if trials else 0,
            'mean_lead_s': round(float(np.mean(leads)), 2) if leads else None,
            'collision_%': round(100 * np.mean([t['collision'] for t in trials]), 1) if trials else 0,
            'mean_min_dist_m': round(float(np.mean([t['min_dist'] for t in trials])), 2) if trials else None}


def table():
    rows = []
    for f in sorted(glob.glob('results/occlusion_*.json')):
        d = json.load(open(f))
        rows.append((d['config'], d['summary']))
    order = list(CONFIGS)
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    print(f"{'config':6} {'trials':>6} {'aware at dart %':>16} {'aware before cross %':>21} {'lead s':>7} {'collision %':>12} {'min dist m':>11}")
    for c, s in rows:
        lead = '-' if s['mean_lead_s'] is None else f"{s['mean_lead_s']:.2f}"
        print(f"{c:6} {s['trials']:6d} {s['aware_at_dart_%']:16.1f} {s['aware_before_crossing_%']:21.1f} {lead:>7} "
              f"{s['collision_%']:12.1f} {s['mean_min_dist_m']:11.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=list(CONFIGS))
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--table', action='store_true')
    args = p.parse_args()
    if args.table:
        return table()

    from simulation.connection import ClientConnection, carla
    from simulation.environment import CarlaEnvironment
    from simulation.vn_traffic import occluded
    from simulation.coop import to_ego
    from bc import expert_action
    share, infra = CONFIGS[args.config]
    client, world = ClientConnection('Town02').setup()
    random.seed(args.seed); np.random.seed(args.seed)
    env = CarlaEnvironment(client, world, 'Town02', checkpoint_frequency=None, traffic=False, share=share, infra=infra)
    trials = []

    def points_on(lidars, loc, r=0.9):
        '''Most LiDAR points any single sensor has on the bike body (0.3-2 m above the road, within r m).'''
        ground = env.map.get_waypoint(loc).transform.location.z  # the bike may still be dropping from its spawn height
        best = 0
        for l in lidars:
            if l.latest is not None:
                q = to_ego(l.latest[0], l.latest[1], np.eye(4))
                h = q[:, 2] - ground
                best = max(best, int(((np.hypot(q[:, 0] - loc.x, q[:, 1] - loc.y) < r) & (h > 0.3) & (h < 2.0)).sum()))
        return best

    try:
        while len(trials) < args.trials:
            random.seed(args.seed * 1000 + len(trials)); np.random.seed(args.seed * 1000 + len(trials))
            if env.reset() is None:
                continue
            done = False
            for _ in range(800):  # drive until a junction (and its RSU) is 40-55 m ahead: the bike stands 7-22 m from the pole
                i = env.current_waypoint_index
                ahead = [k for k, w in enumerate(env.route_waypoints[i:i + 80]) if w.is_junction]
                if ahead and 40 <= ahead[0] <= 55:
                    break
                r = env.step(expert_action(env))
                if r is None or r[2]:
                    done = True
                    break
            if done:
                print('  no junction 40-55 m ahead on this route, new episode', flush=True)
                continue
            bike = occluded(env)
            if not bike:
                print('  scenario did not fit here (no room for the truck / junction too close), new episode', flush=True)
                continue
            # controlled scene: the random coop cars are parked far away (their crossings at the junction caused DENMs and
            # timing changes unrelated to the bike); one observer is parked in the oncoming lane 12 m past the bike
            i = env.current_waypoint_index
            far = sorted(env.map.get_spawn_points(), key=lambda sp: -sp.location.distance(env.vehicle.get_location()))
            for k, l in enumerate(env.lidar_objs[1:]):
                car = l.parent
                car.set_autopilot(False)
                spot = None
                if k == 0 and i + 45 < len(env.route_waypoints):
                    spot = env.route_waypoints[i + 45].get_left_lane()
                tr = spot.transform if spot else far[k]
                car.set_transform(carla.Transform(tr.location + carla.Location(z=0.3), tr.rotation))
                car.apply_control(carla.VehicleControl(hand_brake=True))
            for _ in range(5):
                world.tick()
            hits = []
            col = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=env.vehicle)
            env.sensor_list.append(col)
            col.listen(lambda e: hits.append(e.other_actor.id))
            aware = {'own': None, 'shared': None, 'v2x': None}
            t_cross, min_dist, t = None, 1e9, 0.0
            for step in range(320):
                r = env.step(expert_action(env))
                t += DT
                if r is None or not bike.v.is_alive:
                    break
                b, e = bike.v.get_location(), env.vehicle.get_location()
                min_dist = min(min_dist, b.distance(e))
                if aware['own'] is None and points_on(env.lidar_objs[:1], b) >= 5:
                    aware['own'] = t
                if aware['shared'] is None and points_on(env.lidar_objs, b) >= 5:
                    aware['shared'] = t
                if infra and aware['v2x'] is None and (env.infra.warning_now() is not None or any(
                        math.hypot(o['x'] - b.x, o['y'] - b.y) < 2.0 for o in env.infra.known.values())):
                    aware['v2x'] = t
                # lateral offset of the bike from the ego route
                near = min(env.route_waypoints[env.current_waypoint_index:env.current_waypoint_index + 60],
                           key=lambda w: w.transform.location.distance(b))
                lat = abs((b - near.transform.location).x * near.transform.get_right_vector().x
                          + (b - near.transform.location).y * near.transform.get_right_vector().y)
                if bike.go and 'at_env' not in bike.go:
                    bike.go['at_env'] = t
                if t_cross is None and bike.go and lat < 1.3:  # in the ego's path (car-wide corridor), not just past the lane edge
                    t_cross = t
                if os.environ.get('OCC_TRACE') and step % 5 == 0:
                    ef = env.vehicle.get_transform().get_forward_vector(); dv = b - e
                    w = env.infra.warning if infra else None
                    print(f"   s{step:3d} ego v {env.velocity:4.1f} pedal {expert_action(env)[1]:+.2f} front {min(env.front_obstacle, 99):4.1f} | "
                          f"bike along {dv.x * ef.x + dv.y * ef.y:5.1f} lat {lat:4.1f} v {math.hypot(bike.v.get_velocity().x, bike.v.get_velocity().y) * 3.6:4.1f} go {bool(bike.go)} | "
                          f"warn {env.infra.warning_now() if infra else None} {w and w['kind']} {w and (round(w['x'] - b.x, 1), round(w['y'] - b.y, 1))} | "
                          f"bike riding {bike in env.riders} ctrl {bike.last_control} t {bike.t:.1f}/{bike.life}", flush=True)
                if r[2] or (t_cross is not None and t - t_cross > 4):
                    break
            channels = ['own'] + (['shared'] if share else []) + (['v2x'] if infra else [])
            known = [aware[c] for c in channels if aware[c] is not None]
            t_aware = min(known) if known else None
            t_go = bike.go.get('at_env')
            trial = {'trial': len(trials), 'aware': aware, 't_go': t_go, 't_cross': t_cross, 'lead': lead_time(t_aware, t_go, t_cross),
                     'aware_at_dart': t_aware is not None and t_go is not None and t_aware <= t_go,
                     'collision': bike.v.id in hits, 'min_dist': round(min_dist, 2)}
            trials.append(trial)
            print(f"{args.config} trial {trial['trial']}: lead {trial['lead']} s, collision {trial['collision']}, "
                  f"min dist {trial['min_dist']} m, aware {aware}", flush=True)
            client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list], True)
            env.sensor_list.clear(); env.actor_list.clear(); env.riders.clear(); env.remove_sensors()
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        if env.infra:
            env.infra.destroy(client)
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = False, None; world.apply_settings(s)
    os.makedirs('results', exist_ok=True)
    out = {'config': args.config, 'summary': summarize(trials), 'trials': trials}
    json.dump(out, open(f'results/occlusion_{args.config}.json', 'w'), indent=1)
    print('SUMMARY', args.config, json.dumps(out['summary']))


if __name__ == '__main__':
    main()
