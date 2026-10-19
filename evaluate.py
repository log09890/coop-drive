'''
Fixed-seed evaluation of a driver (PPO/BC policy or the waypoint expert) in the coop env.

  python evaluate.py --town Town02 --vn --name bc_share
  python evaluate.py --town Town02 --vn --no-share --model preTrained_models/ppo_coop_noshare/Town02/bc_policy.pth --name bc_noshare
  python evaluate.py --town Town02 --vn --expert --name expert

Writes results/<name>.json and prints one summary line. Same seeds -> same routes and traffic start.
'''
import argparse
import json
import os
import random
import numpy as np
import torch
from simulation.connection import ClientConnection, carla
from simulation.environment import CarlaEnvironment
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from parameters import LATENT_DIM
from bc import expert_action
from simulation.settings import TM_PORT

DT = 0.05


def summarize(eps):
    '''Totals over episodes; rates per km so short and long episodes weigh fairly.'''
    km = sum(e['km'] for e in eps) or 1e-9
    return {'episodes': len(eps), 'km': round(km, 3),
            'route_done_%': round(100 * np.mean([e['route_done'] for e in eps]), 1),
            'progress_%': round(100 * np.mean([e['progress'] for e in eps]), 1),
            'collisions_per_km': round(sum(e['collisions'] for e in eps) / km, 2),
            'near_miss_s_per_km': round(sum(e['near_miss_steps'] for e in eps) * DT / km, 1),
            'red_light_runs_per_km': round(sum(e['red_runs'] for e in eps) / km, 2),
            'mean_speed_kmh': round(np.mean([e['mean_speed'] for e in eps]), 1),
            'mean_abs_dsteer': round(np.mean([e['jerk'] for e in eps]), 3),
            'ends': {k: sum(e['end'] == k for e in eps) for k in sorted({e['end'] for e in eps})}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--town', default='Town02')
    p.add_argument('--vn', action='store_true')
    p.add_argument('--no-share', action='store_true', help='coop features from the ego LiDAR only')
    p.add_argument('--infra', action='store_true', help='roadside units + V2X feeding the ego')
    p.add_argument('--model', default=None, help='policy .pth (default: latest in the ppo_coop folder)')
    p.add_argument('--expert', action='store_true')
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--steps', type=int, default=1200, help='max steps per episode (0.05 s each)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', required=True)
    args = p.parse_args()

    client, world = ClientConnection(args.town).setup()
    client.get_trafficmanager(TM_PORT).set_random_device_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None, traffic=args.vn, share=not args.no_share, infra=args.infra)
    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, 0.05)
    if not args.expert:
        agent.old_policy.load_state_dict(torch.load(args.model)) if args.model else agent.load()
    eps = []
    try:
        for ep in range(args.episodes):
            random.seed(args.seed * 1000 + ep); np.random.seed(args.seed * 1000 + ep)  # same route + events per episode index
            observation = env.reset()
            if observation is None:
                continue
            hits = []
            col = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=env.vehicle)
            env.sensor_list.append(col)
            col.listen(lambda e: hits.append(e.other_actor.id))
            m = dict(steps=0, near_miss_steps=0, red_runs=0, speed=0.0, km=0.0, dsteer=0.0, end='max_steps')
            was_red = False
            for t in range(args.steps):
                o = encode.process(observation)
                if args.expert:
                    action = expert_action(env)
                else:
                    with torch.no_grad():
                        action = agent.old_policy.actor(o.cpu()).numpy()  # mean action: deterministic evaluation
                result = env.step(action)
                if result is None:
                    m['end'] = 'sim_error'
                    break
                observation, reward, done, info = result
                m['steps'] += 1
                m['speed'] += env.velocity
                m['km'] += env.velocity / 3.6 * DT / 1000
                m['dsteer'] += abs(env.d_steer)
                m['near_miss_steps'] += env.front_obstacle < 2 and env.velocity > 5
                m['red_runs'] += env.red_moving and not was_red
                was_red = env.red_moving
                progress = env.current_waypoint_index / max(1, len(env.route_waypoints) - 1)
                if done:
                    m['end'] = ('collision' if hits else 'off_lane' if env.distance_from_center > env.max_distance_from_center
                                else 'overspeed' if env.velocity > env.max_speed else 'stuck') if reward <= -10 else \
                               'route_done' if progress > 0.98 else 'timeout'
                    break
            if m['end'] == 'max_steps':  # still alive: tidy up like the env does at an episode end
                client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list], True)
                env.sensor_list.clear(); env.actor_list.clear(); env.remove_sensors()
            env.riders.clear()
            n = max(1, m['steps'])
            eps.append({'episode': ep, 'end': m['end'], 'steps': m['steps'], 'km': m['km'], 'progress': progress,
                        'route_done': m['end'] == 'route_done', 'collisions': len(set(hits)),
                        'near_miss_steps': int(m['near_miss_steps']), 'red_runs': int(m['red_runs']),
                        'mean_speed': m['speed'] / n, 'jerk': m['dsteer'] / n})
            print(f'{args.name} ep {ep}: {eps[-1]["end"]}, {m["steps"]} steps, {m["km"] * 1000:.0f} m, '
                  f'collisions {eps[-1]["collisions"]}, near-miss {eps[-1]["near_miss_steps"]} steps', flush=True)
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in env.sensor_list + env.actor_list + env.walker_list], True)
        if env.traffic:
            env.traffic.destroy()
        if env.infra:
            env.infra.destroy(client)
        s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = False, None; world.apply_settings(s)
    os.makedirs('results', exist_ok=True)
    out = {'name': args.name, 'args': vars(args), 'summary': summarize(eps), 'episodes': eps}
    with open(f'results/{args.name}.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('SUMMARY', args.name, json.dumps(out['summary']))


if __name__ == '__main__':
    main()
