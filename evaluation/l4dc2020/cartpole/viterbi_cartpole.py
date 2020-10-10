import os
import torch
import gym
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

import sds_numpy
from sds_numpy.utils import sample_env


rarhmm = torch.load(open('/Users/kek/Documents/informatik/master/semester_3/thesis/code/sds/evaluation/l4dc2020/cartpole/neural_lol_cart.pkl', 'rb'),
                                     map_location='cpu')

env = gym.make('Cartpole-ID-v1')
env._max_episode_steps = 5000
env.unwrapped._dt = 0.01
env.unwrapped._sigma = 1e-4
env.seed(420)

nb_train_rollouts, nb_train_steps = 1, 500
train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)


# plt.figure(figsize=(8, 8))
_, state = rarhmm.viterbi(train_obs, train_act)
_seq = npr.choice(len(train_obs))
_seq = 0


from hips.plotting.colormaps import gradient_cmap
import seaborn as sns


n_plots = env.observation_space.shape[0] + env.action_space.shape[0]
fig, axs = plt.subplots(n_plots, dpi=600)
# fig, axs = plt.subplots(n_plots, figsize=(10,10))


color_names = ["windows blue", "red", "amber", "faded green",
               "dusty purple", "orange", "pale red", "medium green",
               "denim blue", "muted purple"]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

x = np.arange(nb_train_steps)
y_labels = ['x', '$\cos(\\theta)$', '$\sin(\\theta)$', '$\\dot{x}$', '$\\dot{\\theta}$', 'control']
y_lims = [{'low': -5.2, 'high': 5.2}, {'low': -1.2, 'high': 1.2}, {'low': -1.2, 'high': 1.2},
          {'low': -5.2, 'high': 5.2}, {'low': -10.4, 'high': 10.4}, {'low': -5.4, 'high': 5.4}]

for n in range(n_plots - 1):
    axs[n].plot(x, train_obs[0][:, n], color='black')
    axs[n].imshow(state[0][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=[0, nb_train_steps, y_lims[n]['low'], y_lims[n]['high']])
    axs[n].set_xlim(left=0, right=nb_train_steps)
    axs[n].set_ylim(bottom=y_lims[n]['low'], top=y_lims[n]['high'])
    axs[n].set_ylabel(y_labels[n], fontsize=12)
    axs[n].get_yaxis().set_label_coords(-0.08, 0.5)

axs[-1].plot(x, train_act[0], color='black')
axs[-1].set_ylabel(y_labels[-1], fontsize=12)
axs[-1].get_yaxis().set_label_coords(-0.08, 0.5)
axs[-1].imshow(state[0][None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=[0, nb_train_steps, y_lims[-1]['low'], y_lims[-1]['high']])
axs[-1].set_xlim(left=0, right=nb_train_steps)
axs[-1].set_xlabel('steps')

plt.savefig("/Users/kek/Documents/informatik/master/semester_3/thesis/latex/main-thesis/img/switch_dyn_cartpole.pdf",
            format='pdf', bbox_inches = 'tight', dpi = fig.dpi)
plt.show()


#
#
#
# sns.set_style("white")
# sns.set_context("talk")
#
# color_names = ["windows blue", "red", "amber",
#                "faded green", "dusty purple",
#                "orange", "clay", "pink", "greyish",
#                "mint", "light cyan", "steel blue",
#                "forest green", "pastel purple",
#                "salmon", "dark brown"]
#
# colors = sns.xkcd_palette(color_names)
# cmap = gradient_cmap(colors)
#
# plt.subplot(211)
# plt.plot(train_obs[_seq])
# plt.xlim(0, len(train_obs[_seq]))
#
# plt.subplot(212)
# plt.imshow(state[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
# plt.xlim(0, len(train_obs[_seq]))
# plt.ylabel("$z_{\\mathrm{inferred}}$")
# plt.yticks([])
#
# plt.show()