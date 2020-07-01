import gym
import torch
from common.filters import ZFilter
import numpy as np

import matplotlib.pyplot as plt

from sds_torch.rarhmm import rARHMM
import sds_numpy
from lax.a2c_lax import learn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()


# env = gym.make('HybridPendulumTorch-ID-v1')
env = gym.make('Pendulum-ID-v1')
env.unwrapped._dt = 0.01
env.unwrapped._sigma = 1e-4

"""
learn(env, seed=42, obfilter=True, tsteps_per_batch=5000, cv_opt_epochs=5, lax=True, animate=False,
      save_loc='/Users/kek/Documents/informatik/master/semester_3/thesis/code/'
               'sds/evaluation/l4dc2020/pendulum_torch/evals')

"""
# model = torch.load('/Users/kek/Documents/informatik/master/semester_3/thesis/code/'
#                    'sds/evaluation/l4dc2020/pendulum_torch/evals/checkpoint_HybridPendulumTorch-ID-v1_model_.pkl', map_location='cpu')
model = torch.load('/Users/kek/Documents/informatik/master/semester_3/thesis/code/'
                   'sds/evaluation/l4dc2020/pendulum_torch/evals/checkpoint_HybridPendulumTorch-ID-v1_model_low_variance_5.pkl', map_location='cpu')

model.step_policy_model.policy.training = False

obs = env.reset()
obs = to_torch(model.obfilter(obs))
prev_obs = torch.zeros_like(obs)
reward = 0
all_rewards = []
data_obs = []
acts = []
for i in range(50000):
    state = torch.cat([obs, prev_obs], -1)
    prev_obs = torch.clone(obs)
    sampled_u, _, mean, _ = model.step_policy_model.act(state)
    scaled_u = env.action_space.low + (to_npy(sampled_u) + 1.) * 0.5 * (
                  env.action_space.high - env.action_space.low)
    scaled_u = np.clip(scaled_u, a_min=env.action_space.low, a_max=env.action_space.high)
    _obs, r, done, _ = env.step(sampled_u.detach().numpy())
    obs = to_torch(model.obfilter(_obs))
    reward += r
    acts.append(sampled_u.detach())
    # acts.append(scaled_u)
    # print(_obs)
    data_obs.append(_obs)
    if done:
        obs = env.reset()
        obs = to_torch(model.obfilter(obs))
        prev_obs = torch.zeros_like(obs)
        print(reward)
        all_rewards.append(reward)
        reward = 0

print("Expected reward: {} ± {}".format(np.mean(all_rewards), np.std(all_rewards)))

n_plots = env.observation_space.shape[0] + env.action_space.shape[0]
fig, axs = plt.subplots(n_plots)
x = np.arange(len(data_obs))
y_labels = ['$\cos(\\theta)$', '$\sin(\\theta)$', '$\\dot{\\theta}$', 'control']
y_lims = [{'low': -1.2, 'high': 1.2}, {'low': -1.2, 'high': 1.2}, {'low': -8.2, 'high': 8.2}, {'low': -4.4, 'high': 4.4}]
data_obs = np.stack(data_obs)
for n in range(n_plots - 1):
    axs[n].plot(x, data_obs[:,n])
    axs[n].set_ylabel(y_labels[n], fontsize=12)
    axs[n].set_ylim(bottom=y_lims[n]['low'], top=y_lims[n]['high'])

axs[-1].plot(x, acts)
axs[-1].set_ylabel(y_labels[-1], fontsize=12)
axs[-1].set_ylim(bottom=y_lims[-1]['low'], top=y_lims[-1]['high'])
plt.tight_layout()
plt.show()