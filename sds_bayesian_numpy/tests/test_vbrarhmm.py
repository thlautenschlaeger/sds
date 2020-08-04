import gym

from sds_bayesian_numpy.ext.create_gym_data import create_trajectories
from sds_bayesian_numpy.vbrarhmm import VBrARHMM
import numpy as np
import matplotlib.pyplot as plt
from sds_bayesian_numpy.ext.plot import plot_series_prediction, plot_viterbi

from sklearn.metrics import mean_squared_error
import torch

# np.random.seed(1337)
# torch.manual_seed(1337)


# data = np.load('vbhmm/data/hmm_data.npz')
# labels, obs = data['labels'], data['obs']
# obs = [obs]
# data = np.load('vbhmm/data/params_ar.npz', allow_pickle=True)
# sigs,  _obs, labels = data['sigs'], data['obs'], data['labels']
# obs = np.loadtxt('vbhmm/data/data_ball/1/positions.txt')
# _obs = [obs]

env = gym.make('Pendulum-v0')
# env.seed(1337)
# env.spec.max_episode_steps = 5000
# trajectories = create_trajectories(env, [600, 560, 200, 150])
trajectories = create_trajectories(gym.make('Pendulum-v0'), [200, 50, 200, 150, 150, 120, 120, 178, 105, 170, 200])
_obs = []
actions = []

for trajetory in trajectories:
    trajetory['obs'] = trajetory['obs']
    _obs.append(trajetory['obs'])
    actions.append(trajetory['actions'])

n_predictions = 100
traj = create_trajectories(env, [200])
# traj[0]['obs'] = traj[0]['obs'] * 10000
hist_obs = [traj[0]['obs'][:-n_predictions]]
hist_act = [traj[0]['actions'][:-n_predictions]]
nxt_act = [traj[0]['actions'][n_predictions:]]

# obs = [_obs[0][:-n_predictions]]

# obs = _obs
# np.random.seed(1337)
num_states = 5
obs_dim = hist_obs[0].shape[1]
act_dim = actions[0].shape[1]
ar_steps = 1


# Set priors
init_prior = {'omega0': np.ones(num_states) / num_states}
trans_prior = None
observation_prior = {'W0': np.eye(obs_dim),
                     'nu0': obs_dim + 2,
                     'm0': np.zeros(obs_dim),
                     'beta0': 0.05,
                     'P0':np.eye(obs_dim),
                     'eta0': obs_dim + 2,
                     'K0': np.eye((obs_dim + act_dim) + 1) * 0.25,
                     'M0': np.zeros((obs_dim, (obs_dim + act_dim) + 1))
                     # 'M0': np.random.multivariate_normal(np.zeros(obs_dim + 1), np.eye(obs_dim + 1))
                     }

vbrarhmm = VBrARHMM(obs=_obs, act=actions, n_states=num_states, init_prior=init_prior,
                  trans_prior=trans_prior, obs_prior=observation_prior)

log_liks = []
log_liks = vbrarhmm.em(_obs, act=actions, n_iter=20, prec=10e-4)

delta, predicted_sequence = vbrarhmm.viterbi(_obs)

# horizon = [901]
# state, obs_ = vbarhmm.sample(horizon)

next_state, next_obs = vbrarhmm.forecast(hist_obs, hist_act=hist_act, nxt_act=nxt_act, horizon=[n_predictions], stoch=False, average=False,
                                         stoch_reps=1)
# print("Next state:", next_state[0])
# print("Next obs:", next_obs[0])
# print("True next obs:", [_obs[0][-n_predictions-1:]])

# true = _obs[0][-n_predictions-1:]
true_trajectory = traj[0]['obs'][-n_predictions - 1:]
prediction = next_obs[0]

# plot_viterbi([labels[0][:-n_predictions]], predicted_sequence)
# plot_viterbi([labels[0]], predicted_sequence)
plot_viterbi(predicted_sequence, predicted_sequence)

error = mean_squared_error(true_trajectory, prediction)

# print("Predicted: \n", predicted_sequence[0])
# print("True: \n", labels[0])
fig = plot_series_prediction(true_trajectory, prediction, title="Mean prediction")
#
# plt.plot(log_liks)
plt.show()
