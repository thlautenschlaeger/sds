import gym
import torch
import numpy as np
import seaborn as sns
from hips.plotting.colormaps import gradient_cmap
import matplotlib.pyplot as plt
import os
from tikzplotlib import save

from sds_numpy import rARHMM
from sds_torch.rarhmm import rARHMM
from lax.a2c_lax import learn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()

# env = gym.make('Cartpole-ID-v1') # <--- eval on cartpole
env = gym.make('HybridCartpole-ID-v1') # <--- train on hybrid cartpole

env.unwrapped._dt = 0.01
env.unwrapped._sigma = 1e-4
env._max_episode_steps = 5000

"""
learn(env, seed=42, obfilter=True, tsteps_per_batch=5000, cv_opt_epochs=5, lax=False, animate=False,
      gamma=0.99, vf_opt_epochs=50, total_steps=int(50e6),
      save_loc='/Users/kek/Documents/informatik/master/semester_3/thesis/code/'
               'sds/evaluation/l4dc2020/cartpole/evals')
"""

model = torch.load('/Users/kek/Documents/informatik/master/semester_3/thesis/code/sds/evaluation/l4dc2020/cartpole/thesis_eval/checkpoint_HybridCartpole-ID-v1_model_887_epochs_.pkl', map_location='cpu')

model.step_policy_model.policy.training = False

seed = 100
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

obs = env.reset()
obs = to_torch(model.obfilter(obs))
prev_obs = torch.zeros_like(obs)
reward = 0
all_rewards = []
env_obs = []
env_acts = []
horizon = 100000
for i in range(horizon):
    identified_states = torch.cat([obs, prev_obs], -1)
    prev_obs = torch.clone(obs)
    sampled_u, _, mean, _ = model.step_policy_model.act(identified_states)
    scaled_u = env.action_space.low + (to_npy(sampled_u) + 1.) * 0.5 * (
                  env.action_space.high - env.action_space.low)
    scaled_u = np.clip(scaled_u, a_min=env.action_space.low, a_max=env.action_space.high)
    _obs, r, done, _ = env.step(scaled_u)
    obs = to_torch(model.obfilter(_obs))
    reward += r
    env_acts.append(sampled_u.detach())
    # acts.append(scaled_u)
    # print(i, _obs)
    env_obs.append(to_torch(_obs))
    if done:
        obs = env.reset()
        obs = to_torch(model.obfilter(obs))
        prev_obs = torch.zeros_like(obs)
        print(reward)
        all_rewards.append(reward)
        reward = 0

print("Expected reward: {} ± {}".format(np.mean(all_rewards), np.std(all_rewards)))
"""
rarhmm = torch.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..' ))
                                          + '/sds_numpy/envs/hybrid/models/neural_rarhmm_cartpole_cart.pkl', 'rb'),
                                     map_location='cpu')


_, identified_states = rarhmm.viterbi([np.stack([to_npy(o) for o in env_obs])], [np.stack([to_npy(a) for a in env_acts])])
# rarhmm.viterbi([to_npy(env_obs[i][None]) for i in range(500)], [to_npy(env_acts[i][None]) for i in range(500)])
color_names = ["windows blue", "red", "amber", "faded green",
               "dusty purple", "orange", "pale red", "medium green",
               "denim blue", "muted purple"]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

identified_states = [np.stack(identified_states).squeeze()]
n_plots = env.observation_space.shape[0] + env.action_space.shape[0]
fig, axs = plt.subplots(n_plots)
x = np.arange(len(env_obs))
y_labels = ['x', '$\cos(\\theta)$', '$\sin(\\theta)$', '$\\dot{x}$', '$\\dot{\\theta}$', 'control']
y_lims = [{'low': -5.2, 'high': 5.2}, {'low': -1.5, 'high': 1.5}, {'low': -1.2, 'high': 1.2},
          {'low': -5.2, 'high': 5.2}, {'low': -11.8, 'high': 11.8}, {'low': -5.4, 'high': 5.4}]

env_obs = np.stack(env_obs)
for n in range(n_plots - 1):
    axs[n].plot(x, env_obs[:, n], color='black')
    axs[n].imshow(identified_states[0][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=[0, horizon, y_lims[n]['low'], y_lims[n]['high']])
    axs[n].set_ylabel(y_labels[n], fontsize=12)
    axs[n].set_ylim(bottom=y_lims[n]['low'], top=y_lims[n]['high'])

axs[-1].plot(x, env_acts, color='black')
axs[-1].set_ylabel(y_labels[-1], fontsize=12)
axs[-1].imshow(identified_states[0][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=[0, horizon, y_lims[-1]['low'], y_lims[-1]['high']])
axs[-1].set_ylim(bottom=y_lims[-1]['low'], top=y_lims[-1]['high'])
axs[-1].set_xlim(left=0, right=horizon)
axs[-1].set_xlabel('steps')
plt.tight_layout()
save('cartpole-policy-rarhmm-dynamics.tex', externalize_tables=True)
plt.show()
"""