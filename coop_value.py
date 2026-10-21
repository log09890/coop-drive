'''
How much does cooperative perception help a learned driver? An offline, sim-free experiment on the expert dataset.

Four input configurations, same data, same network and training (behavior cloning of the expert's actions):
  ego   own LiDAR only (obs_alt), infrastructure inputs zeroed
  v2v   + shared LiDAR of the coop cars (obs), infrastructure zeroed
  v2i   own LiDAR (obs_alt) + infrastructure (RSU objects, SPaT, DENM)
  full  everything (obs)
Episodes (not steps) are split into train / held-out, repeated over seeds. Measured on the held-out episodes:
  action error overall and on the critical steps (where the shared view differs from the ego view, or a DENM is on),
  hard-brake recall / precision, and how early the model starts braking before the expert's hard brake.
Also reports how often the cooperative inputs reveal something the ego's own LiDAR does not.

  python coop_value.py [data/bc_Town02.npz] [--seeds 3] [--epochs 15]   -> results/coop_value.json + a table
'''
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

DT = 0.05
HARD, PRED_BRAKE = -0.5, -0.3  # expert pedal <= HARD is a hard brake; the model "brakes" when its pedal <= PRED_BRAKE


def layout(dim):
    '''Index ranges of the observation parts for the 158 (no signals) and 177 (with signals) layouts.'''
    infra = 24
    sig = dim - 100 - 34 - infra  # 0 or 19
    assert sig in (0, 19), dim
    return {'corridor': 116, 'infra': slice(100 + 34 + sig, dim), 'denm': 100 + 34 + sig + 21}


def variants(obs, alt, lay):
    no_infra = lambda x: np.concatenate([x[:, :lay['infra'].start], np.zeros_like(x[:, lay['infra']])], 1)
    return {'ego': no_infra(alt), 'v2v': no_infra(obs), 'v2i': alt, 'full': obs}


def episodes(done):
    ends = np.flatnonzero(done) + 1
    starts = np.r_[0, ends[:-1]]
    return [np.arange(a, b) for a, b in zip(starts, ends) if b > a]


def actor(dim, adim):
    return nn.Sequential(nn.Linear(dim, 500), nn.Tanh(), nn.Linear(500, 300), nn.Tanh(), nn.Linear(300, 100), nn.Tanh(),
                         nn.Linear(100, adim), nn.Tanh())  # = the PPO actor


def fit(x, y, epochs, seed, dev):
    torch.manual_seed(seed)
    net = actor(x.shape[1], y.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, y = torch.tensor(x, device=dev), torch.tensor(y, device=dev)
    for _ in range(epochs):
        for b in torch.randperm(len(x), device=dev).split(256):
            loss = ((net(x[b]) - y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def anticipation(pred, act, eps, window=40):
    '''Mean seconds before each expert hard-brake onset at which the model is already braking (0 if only at the onset).'''
    leads, missed = [], 0
    for ep in eps:
        p, a = pred[ep], act[ep]
        for t in range(1, len(ep)):
            if a[t] <= HARD and a[t - 1] > -0.1:  # onset
                k = next((k for k in range(max(0, t - window), t + 1) if p[k] <= PRED_BRAKE), None)
                if k is None:
                    missed += 1
                else:
                    leads.append((t - k) * DT)
    return (float(np.mean(leads)) if leads else 0.0), missed, len(leads) + missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', nargs='?', default='data/bc_Town02.npz')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()
    z = np.load(args.data)
    obs, alt, act = z['obs'].astype(np.float32), z['obs_alt'].astype(np.float32), z['act'].astype(np.float32)
    lay = layout(obs.shape[1])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # what the cooperative inputs add, in the data itself
    own_clear = alt[:, lay['corridor']] >= 0.999
    hidden = (obs[:, lay['corridor']] < 0.999) & own_clear  # shared LiDAR sees an obstacle in the lane, own LiDAR does not
    closer = obs[:, lay['corridor']] < alt[:, lay['corridor']] - 1e-3  # shared view puts the obstacle closer
    denm = obs[:, lay['denm']] > 0.5
    critical = closer | denm
    info = {'steps': int(len(obs)), 'episodes': int(z['done'].sum()), 'hidden_obstacle_%': round(100 * hidden.mean(), 2),
            'shared_obstacle_closer_%': round(100 * closer.mean(), 2), 'denm_active_%': round(100 * denm.mean(), 2),
            'critical_%': round(100 * critical.mean(), 2), 'expert_hard_brake_%': round(100 * (act[:, 1] <= HARD).mean(), 2)}

    eps = episodes(z['done'])
    X = variants(obs, alt, lay)
    res = {k: [] for k in X}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(eps))
        n_val = max(1, len(eps) // 5)
        val_eps = [eps[i] for i in order[:n_val]]
        tr = np.concatenate([eps[i] for i in order[n_val:]])
        va = np.concatenate(val_eps)
        for name, x in X.items():
            net = fit(x[tr], act[tr], args.epochs, seed, dev)
            with torch.no_grad():
                pred = net(torch.tensor(x, device=dev)).cpu().numpy()
            pv, av = pred[va], act[va]
            hard = av[:, 1] <= HARD
            said = pv[:, 1] <= PRED_BRAKE
            crit = critical[va]
            lead, missed, onsets = anticipation(pred[:, 1], act[:, 1], val_eps)
            res[name].append({'mse': float(((pv - av) ** 2).mean()), 'pedal_mse': float(((pv[:, 1] - av[:, 1]) ** 2).mean()),
                              'critical_pedal_mse': float(((pv[crit, 1] - av[crit, 1]) ** 2).mean()) if crit.any() else None,
                              'brake_recall': float(said[hard].mean()) if hard.any() else None,
                              'brake_precision': float(hard[said].mean()) if said.any() else None,
                              'brake_lead_s': lead, 'missed_onsets': missed, 'onsets': onsets})
            print(f'seed {seed} {name:4}: pedal mse {res[name][-1]["pedal_mse"]:.4f}  critical {res[name][-1]["critical_pedal_mse"]}  '
                  f'recall {res[name][-1]["brake_recall"]}  lead {lead:.2f} s', flush=True)

    def agg(name, key):
        v = [r[key] for r in res[name] if r[key] is not None]
        return (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) if v else (None, None)
    summary = {name: {k: agg(name, k) for k in ('pedal_mse', 'critical_pedal_mse', 'brake_recall', 'brake_precision', 'brake_lead_s')}
               for name in res}
    os.makedirs('results', exist_ok=True)
    json.dump({'data': args.data, 'info': info, 'summary': summary, 'runs': res, 'seeds': args.seeds, 'epochs': args.epochs},
              open('results/coop_value.json', 'w'), indent=1)
    print('\nINFO', json.dumps(info))
    print(f"{'config':6} {'pedal MSE':>16} {'critical pedal MSE':>20} {'brake recall':>16} {'brake precision':>16} {'brake lead s':>14}")
    fmt = lambda m: '-' if m[0] is None else f'{m[0]:.3f}±{m[1]:.3f}'
    for name, s in summary.items():
        print(f"{name:6} {fmt(s['pedal_mse']):>16} {fmt(s['critical_pedal_mse']):>20} {fmt(s['brake_recall']):>16} "
              f"{fmt(s['brake_precision']):>16} {fmt(s['brake_lead_s']):>14}")


if __name__ == '__main__':
    main()
