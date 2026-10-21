import os
import numpy as np

import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import  *

device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        
        #self.env = env
        self.obs_dim = 100 + COOP_OBS
        self.action_dim = 3 if COOP_OBS else 2  # coop: steer, throttle/brake, horn
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.checkpoint_file_no = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs, train):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))
        if train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std


    def learn(self, epochs=10, minibatch=512, bc_actor=None, bc_coef=0.0, max_grad_norm=0.5):
        '''
        PPO update on the rollout in memory. Returns the update's statistics.
        Upstream did 7 full-batch steps per update (~20k samples): almost no learning per update. Here: `epochs`
        passes of shuffled minibatches, advantages normalized per update, gradient norm clipped, and optionally
        a behavior-cloning anchor bc_coef * ||mu(s) - mu_bc(s)||^2 that keeps the fine-tuned policy near the BC policy.
        Value targets stay the normalized Monte Carlo returns (what bc.py pre-trains the critic on).
        '''
        returns, g = [], 0.0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            g = reward + self.gamma * g * (not is_terminal)
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)
        with torch.no_grad():
            adv = returns - self.policy.critic(old_states).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            bc_target = bc_actor(old_states) if bc_actor is not None and bc_coef > 0 else None

        n, stats = len(returns), {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0,
                                  'clip_frac': 0.0, 'bc_loss': 0.0, 'grad_norm': 0.0}
        steps = 0
        for _ in range(epochs):
            for idx in torch.randperm(n).split(minibatch):
                logprobs, values, entropy = self.policy.evaluate(old_states[idx], old_actions[idx])
                ratio = torch.exp(logprobs - old_logprobs[idx])
                surr = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[idx])
                policy_loss = -surr.mean()
                value_loss = self.MseLoss(values.squeeze(-1), returns[idx])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                bc_loss = torch.zeros(())
                if bc_target is not None:
                    bc_loss = ((self.policy.actor(old_states[idx]) - bc_target[idx]) ** 2).sum(-1).mean()
                    loss = loss + bc_coef * bc_loss
                self.optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    for k, v in (('policy_loss', policy_loss), ('value_loss', value_loss), ('entropy', entropy.mean()),
                                 ('approx_kl', (old_logprobs[idx] - logprobs).mean()), ('bc_loss', bc_loss), ('grad_norm', gn),
                                 ('clip_frac', ((ratio - 1).abs() > self.clip).float().mean())):
                        stats[k] += float(v)
                steps += 1
        stats = {k: v / max(1, steps) for k, v in stats.items()}
        stats.update(samples=n, sgd_steps=steps, mean_reward=float(np.mean(self.memory.rewards)) if n else 0.0)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()
        return stats

    
    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2]) - 1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))
            