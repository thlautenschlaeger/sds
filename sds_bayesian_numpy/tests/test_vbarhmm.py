from vbhmm.hmm.vbarhmm import VBARHMM
import numpy as np
import matplotlib.pyplot as plt
from vbhmm.ext.plot import plot_series_prediction, plot_viterbi


# data = np.load('vbhmm/data/hmm_data.npz')
# labels, obs = data['labels'], data['obs']
# obs = [obs]
data = np.load('vbhmm/data/params_ar.npz', allow_pickle=True)
sigs,  _obs, labels = data['sigs'], data['obs'], data['labels']
# obs = np.loadtxt('vbhmm/data/data_ball/1/positions.txt')
# obs = [obs]

n_predictions = 20

obs = [_obs[0][:-n_predictions]]

# obs = _obs
# np.random.seed(1337)
num_states = 5
obs_dim = obs[0].shape[1]
ar_steps = 1


# Set priors
init_prior = {'omega0': np.ones(num_states) / num_states}
transition_prior = {'omega0': np.ones((num_states, num_states)) / num_states}
observation_prior = {'W0': np.eye(obs_dim),
                     'nu0': 4,
                     'm0': np.zeros(obs_dim),
                     'beta0': 0.05,
                     'P0':np.eye(obs_dim),
                     'eta0': 5,
                     'K0': np.eye((obs_dim * ar_steps) + 1),
                     'M0': np.zeros((obs_dim, (obs_dim * ar_steps) + 1))
                     # 'M0': np.random.multivariate_normal(np.zeros(obs_dim + 1), np.eye(obs_dim + 1))
                     }

vbarhmm = VBARHMM(obs=obs, act={}, n_states=num_states, init_prior=init_prior,
                  trans_prior=transition_prior, obs_prior=observation_prior)

log_liks = []
log_liks = vbarhmm.em(obs,n_iter=50, prec=10e-4)

delta, predicted_sequence = vbarhmm.viterbi(obs)

# horizon = [901]
# state, obs_ = vbarhmm.sample(horizon)

next_state, next_obs = vbarhmm.forecast([obs[0]], horizon=[n_predictions], stoch=False, average=False,
                                        stoch_reps=3)
# print("Next state:", next_state[0])
# print("Next obs:", next_obs[0])
# print("True next obs:", [_obs[0][-n_predictions-1:]])

true = _obs[0][-n_predictions-1:]
prediction = next_obs[0]

plot_series_prediction(true,prediction, title="Mean prediction")
# plot_viterbi(labels, predicted_sequence)


print("Predicted: \n", predicted_sequence[0])
print("True: \n", labels[0])
#
# plt.plot(log_liks)
plt.show()
