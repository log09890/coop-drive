'''
Analytics behind the dashboard, numpy/torch only (no CARLA): conflict prediction, input sensitivity, session metrics.
'''
import numpy as np
import torch

DT = 0.05


from simulation.infrastructure import closest_approach  # noqa: F401  (shared with the edge server)


def input_sensitivity(actor, obs):
    '''
    How much each input moves the policy right now: |d steer / d x| and |d pedal / d x| for every observation value.
    actor: the PPO actor network, obs: 1-D observation tensor. Returns two numpy arrays shaped like obs.
    '''
    x = obs.detach().cpu().float().clone().requires_grad_(True)
    out = actor(x.unsqueeze(0))[0]
    steer, = torch.autograd.grad(out[0], x, retain_graph=True)
    pedal, = torch.autograd.grad(out[1], x)
    return steer.abs().numpy(), pedal.abs().numpy()


def group_sensitivity(s, latent, n_sectors):
    '''Sum one sensitivity vector into readable input groups (layout: latent | 5 navigation | sectors, corridor, deltas).'''
    nav = ['throttle', 'speed', 'speed/target', 'lane offset', 'heading']
    g = {'camera (VAE)': s[:latent].sum()}
    for k, name in enumerate(nav):
        g[name] = s[latent + k]
    c = latent + 5
    g['LiDAR sectors'] = s[c:c + n_sectors].sum()
    g['lane corridor'] = s[c + n_sectors]
    g['motion (deltas)'] = s[c + n_sectors + 1:c + 2 * (n_sectors + 1)].sum()
    rest = s[c + 2 * (n_sectors + 1):]  # infrastructure part, if the observation has one
    if len(rest):
        g['V2X objects'] = rest[:-8].sum()
        g['traffic light (SPaT)'] = rest[-8:-3].sum()
        g['DENM warning'] = rest[-3:-1].sum()
    return g


class SessionMetrics:
    '''Running totals over a whole dashboard session, same definitions as evaluate.py.'''

    def __init__(self):
        self.km = self.near_miss_s = self.speed_sum = 0.0
        self.steps = self.collisions = self.red_runs = self.episodes = 0
        self.was_red = False

    def step(self, speed_kmh, front_m, red_moving):
        '''Returns the events of this step: 'near_miss' and/or 'red_light'.'''
        self.steps += 1
        self.speed_sum += speed_kmh
        self.km += speed_kmh / 3.6 * DT / 1000
        events = []
        if front_m < 2 and speed_kmh > 5:
            self.near_miss_s += DT
            events.append('near_miss')
        if red_moving and not self.was_red:
            self.red_runs += 1
            events.append('red_light')
        self.was_red = red_moving
        return events

    def summary(self):
        km = max(self.km, 1e-9)
        return {'km': self.km, 'collisions_per_km': self.collisions / km, 'near_miss_s_per_km': self.near_miss_s / km,
                'red_runs_per_km': self.red_runs / km, 'mean_speed': self.speed_sum / max(1, self.steps)}
