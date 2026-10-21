'''
Behavior cloning warm start for the coop PPO agent.

  python bc.py collect --steps 40000   # expert follows the route waypoints, records (obs, action)
  python bc.py train                   # fit actor (and critic) on it, write it as the latest PPO checkpoint
  python continuous_driver.py --exp-name ppo --load-checkpoint True   # PPO fine-tune from the BC policy
'''
import argparse
import math
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
from simulation.connection import carla
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import LATENT_DIM, GAMMA, COOP_OBS, PPO_CHECKPOINT_DIR

DATA = 'data/bc_{}.npz'


def expert_action(env, lookahead=6):
    '''Pure pursuit on the route + stop for what the LiDAR sees in the lane; action in PPO space (steer, throttle>0 / brake<0).'''
    wps = env.route_waypoints
    target = wps[min(env.current_waypoint_index + lookahead, len(wps) - 1)].transform.location
    t = env.vehicle.get_transform()
    fwd = t.get_forward_vector()
    angle = math.atan2(target.y - t.location.y, target.x - t.location.x) - math.atan2(fwd.y, fwd.x)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi  # left-handed UE4: positive = target on the right = steer right
    steer = float(np.clip(1.5 * angle, -1, 1))
    # reward peaks at 15-22 km/h (>25 ends the episode). Something in the lane corridor: follow it and stop ~3.5 m behind
    # if it stays; a bike just crossing (in and out within CROSS_S) only makes us ease off, it does not stop the car.
    # shortcut: the corridor is straight ahead, so in sharp curves it can miss/catch things; use the route waypoints if that matters
    front, now = env.front_obstacle, env.now()
    since = getattr(env, 'blocked_since', None)
    env.blocked_since = (now if since is None or since > now else since) if front < 8.0 else None  # since > now: new episode
    lingering = env.blocked_since is not None and now - env.blocked_since >= CROSS_S
    follow = (front - 3.5) * 2.0
    target_speed = min(20.0, max(0.0, follow if lingering else max(follow, min(8.0, (front - STOP_GAP) * 3.0))))
    urgent = front < STOP_GAP
    if env.vehicle.is_at_traffic_light() and env.vehicle.get_traffic_light_state() == carla.TrafficLightState.Red:
        target_speed = 0.0  # only reachable in --vn mode, elsewhere the env turns the ego's lights green
    warn = env.infra.warning_now() if getattr(env, 'infra', None) else None
    if warn is not None:  # DENM from the junction edge server: someone will cut our path in `warn` s
        urgent = urgent or warn < 1.0
        target_speed = min(target_speed, 0.0 if warn < 1.0 else 10.0)
    if urgent:
        pedal = -1.0
    elif target_speed < 1 and env.velocity < 1:
        pedal = -0.5  # hold at a standstill
    elif env.velocity > target_speed + 1:
        pedal = -float(np.clip(0.08 * (env.velocity - target_speed), 0.15, 1))  # graded: hard only when far too fast
    else:
        pedal = float(np.clip(0.4 + 0.15 * (target_speed - env.velocity), 0, 1))
    # horn, Vietnamese style: a short tap every 2 s while someone dawdles in the lane close ahead, or on a DENM
    honk = -1.0
    if (lingering and front < 8.0 and env.velocity < 10) or warn is not None:
        if now - getattr(env, 'last_honk', -9.0) > 2.0 or now < getattr(env, 'last_honk', -9.0):
            env.last_honk = now
        honk = 1.0 if now - env.last_honk < 0.3 else -1.0
    return np.array([steer, pedal, honk])


STOP_GAP = 2.5  # m: closer than this in the lane -> full brake at once
CROSS_S = 0.3   # s an obstacle must stay in the lane corridor before the expert stops for it


def collect(args):
    from simulation.connection import ClientConnection
    from simulation.environment import CarlaEnvironment
    from simulation.scenarios import spawn_scenario
    client, world = ClientConnection(args.town).setup()
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None, traffic=args.vn, infra=args.infra)
    encode = EncodeState(LATENT_DIM)
    obs_l, alt_l, act_l, rew_l, done_l = [], [], [], [], []
    while len(obs_l) < args.steps:
        observation = env.reset()
        if observation is None:
            continue
        ep_reward, done, stuck = 0.0, False, 0
        alt = env.coop_obs_other()  # the same moment seen without sharing (ego LiDAR only), for the ablation
        while not done and len(obs_l) < args.steps:
            o = encode.process(observation).cpu().numpy()
            o_alt = np.concatenate([o[:100], alt])
            a = expert_action(env)
            noisy = a + np.r_[np.random.normal(0, args.noise), np.zeros(len(a) - 1)]  # execute noisy, label clean: data also covers recoveries
            result = env.step(noisy)
            if result is None:  # env crashed and cleaned up, start a new episode
                break
            observation, reward, done, info = result
            # traffic to brake for (pedestrian crossing or slow car ahead), only on a clear road so obstacles do not pile up
            if args.scenarios and env.timesteps % 250 == 0 and env.front_obstacle == np.inf and env.velocity > 10:
                spawn_scenario(env, random.choice([1, 2]))
            stuck = stuck + 1 if env.velocity < 1 else 0
            if stuck > 400:  # 20 s still (e.g. stuck traffic): give up this episode
                done = True
            if not done:
                alt = env.coop_obs_other()
            obs_l.append(o); alt_l.append(o_alt); act_l.append(a); rew_l.append(reward); done_l.append(done)
            ep_reward += reward
        if done_l:
            done_l[-1] = True  # cut an unfinished episode for the return computation
        save(args.town, obs_l, alt_l, act_l, rew_l, done_l)  # every episode, so stopping early keeps the data
        print(f'collected {len(obs_l)} steps (saved), episode reward {ep_reward:.1f}, distance {info[0] if done else "-"}', flush=True)


def save(town, obs_l, alt_l, act_l, rew_l, done_l):
    os.makedirs('data', exist_ok=True)
    np.savez_compressed(DATA.format(town), obs=np.array(obs_l, np.float32), obs_alt=np.array(alt_l, np.float32),
                        act=np.array(act_l, np.float32),
                        rew=np.array(rew_l, np.float32), done=np.array(done_l))


def discounted_returns(rew, done, gamma):
    '''Same Monte Carlo targets PPOAgent.learn uses for its critic (before normalization).'''
    out, g = np.zeros_like(rew), 0.0
    for i in reversed(range(len(rew))):
        if done[i]:
            g = 0.0
        g = rew[i] + gamma * g
        out[i] = g
    return out


def train(args):
    d = np.load(DATA.format(args.town))
    obs, act = torch.tensor(d['obs_alt' if args.noshare else 'obs']), torch.tensor(d['act'])
    ret = discounted_returns(d['rew'], d['done'], GAMMA)
    ret = torch.tensor((ret - ret.mean()) / (ret.std() + 1e-7))  # PPO normalizes its critic targets the same way
    assert obs.shape[1] == 100 + COOP_OBS, 'data was collected with another COOP_SECTORS/COOP_OBS, collect again'
    policy = ActorCritic(obs.shape[1], act.shape[1], args.action_std)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    idx = torch.randperm(len(obs))
    val, tr = idx[:len(idx) // 10], idx[len(idx) // 10:]
    mse = nn.MSELoss()
    for epoch in range(args.epochs):
        for b in tr[torch.randperm(len(tr))].split(256):
            loss = mse(policy.actor(obs[b]), act[b]) + 0.5 * mse(policy.critic(obs[b]).squeeze(-1), ret[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                print(f'epoch {epoch}: val actor mse {mse(policy.actor(obs[val]), act[val]):.4f}, '
                      f'val critic mse {mse(policy.critic(obs[val]).squeeze(-1), ret[val]):.4f}', flush=True)
    if args.noshare:  # ablation model: evaluated with `evaluate.py --no-share`, never fine-tuned
        os.makedirs(f'preTrained_models/ppo_coop_noshare/{args.town}', exist_ok=True)
        torch.save(policy.state_dict(), f'preTrained_models/ppo_coop_noshare/{args.town}/bc_policy.pth')
        return print(f'saved no-share BC policy to preTrained_models/ppo_coop_noshare/{args.town}/bc_policy.pth')
    # Become the only/latest checkpoint so `--load-checkpoint True` / `--train False` pick it up
    model_dir, meta_dir = PPO_CHECKPOINT_DIR + args.town, f'checkpoints/PPO_coop/{args.town}'
    os.makedirs(model_dir, exist_ok=True); os.makedirs(meta_dir, exist_ok=True)
    for d in (model_dir, meta_dir):  # older models / training counters: moved aside, never overwritten
        if os.listdir(d):
            old = f"{d}_backup/{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(old)
            for f in os.listdir(d):
                os.replace(os.path.join(d, f), os.path.join(old, f))
            print(f'moved the previous contents of {d} to {old}')
    torch.save(policy.state_dict(), f'{model_dir}/ppo_policy_0_.pth')
    torch.save(policy.state_dict(), f'data/bc_policy_{args.town}.pth')  # PPO overwrites ppo_policy_0 later, keep a copy
    with open(f'{meta_dir}/checkpoint_ppo_0.pickle', 'wb') as f:
        pickle.dump({'cumulative_score': 0, 'episode': 0, 'timestep': 0, 'action_std_init': args.action_std}, f)
    print('saved BC policy to', model_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['collect', 'train'])
    p.add_argument('--town', default='Town07')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--noise', type=float, default=0.2, help='steer noise std while collecting')
    p.add_argument('--vn', action='store_true', help='collect in chaotic Vietnamese-style traffic (use with --town Town02)')
    p.add_argument('--infra', action='store_true', help='collect with roadside units + V2X in the observation')
    p.add_argument('--noshare', action='store_true', help='train: use the ego-only LiDAR observations (ablation)')
    p.add_argument('--no-scenarios', dest='scenarios', action='store_false', help='no pedestrians / slow cars while collecting')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--action-std', type=float, default=0.05, help='PPO exploration for the fine-tune, small to keep the BC behaviour')
    args = p.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    collect(args) if args.mode == 'collect' else train(args)
