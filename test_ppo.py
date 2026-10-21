# python test_ppo.py  (no CARLA): the PPO update learns a bandit, and the BC anchor holds the policy in place
import copy
import torch
from networks.on_policy.ppo.agent import PPOAgent

torch.manual_seed(0)
TARGET = torch.tensor([0.5, -0.3, -0.8])[:PPOAgent('Town02', 0.05).action_dim]


def rollout(agent, n=2048):
    for _ in range(n):  # one-step episodes, reward peaks at TARGET
        s = torch.randn(agent.obs_dim)
        a = torch.tensor(agent.get_action(s, train=True))
        agent.memory.rewards.append(float(-((a - TARGET) ** 2).sum()))
        agent.memory.dones.append(True)


def mean_action(agent):
    with torch.no_grad():
        return agent.old_policy.actor(torch.randn(256, agent.obs_dim)).mean(0)


agent = PPOAgent('Town02', 0.05)
start = (mean_action(agent) - TARGET).norm()
for it in range(8):
    rollout(agent)
    stats = agent.learn()
end = (mean_action(agent) - TARGET).norm()
assert end < 0.5 * start, (float(start), float(end))
assert stats['sgd_steps'] == 10 * 4 and 0 <= stats['clip_frac'] <= 1 and abs(stats['approx_kl']) < 0.2, stats

# a strong BC anchor keeps the actor where it started
anchored = PPOAgent('Town02', 0.05)
bc = copy.deepcopy(anchored.policy.actor)
before = mean_action(anchored)
for it in range(4):
    rollout(anchored)
    s = anchored.learn(bc_actor=bc, bc_coef=50.0)
moved = (mean_action(anchored) - before).norm()
assert moved < 0.05 and s['bc_loss'] < 1e-2, (float(moved), s)
print('ok', f'distance to target {float(start):.3f} -> {float(end):.3f}; anchored drift {float(moved):.4f}')
