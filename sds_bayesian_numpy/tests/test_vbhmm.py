from vbhmm.hmm.vbhmm import VBHMM
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys
sys.path.append('.')


# data = np.load('vbhmm/data/hmm_data.npz')
# labels, obs = data['labels'], data['obs']
# obs = [obs]
data = np.load('vbhmm/data/params.npz', allow_pickle=True)
sigs, mus, obs, labels = data['sigs'], data['mus'], data['obs'], data['labels']

# obs = np.loadtxt('vbhmm/data/data_ball/1/positions.txt')
# obs = [obs]
# np.random.seed(133232)
np.random.seed(1336)


num_states = 5
obs_dim = obs[0].shape[1]

# Set priors
init_prior = {'omega0': np.ones(num_states) / num_states}
transition_prior = {'omega0': np.ones((num_states, num_states)) / num_states}
observation_prior = {'W0': np.eye((obs_dim)),
                     'nu0': 3,
                     'm0': np.zeros(obs_dim),
                     'beta0': 0.05}

vbhmm = VBHMM(obs=list(obs), n_states=num_states, act={}, init_prior=init_prior,
              trans_prior=transition_prior, obs_prior=observation_prior)

log_liks = []
start = time()
log_liks = vbhmm.em(list(obs), n_iter=100)
print("Duration: {}".format( time() - start))

pi = np.exp(vbhmm.init_model.log_prob)
trans_prob = np.exp(vbhmm.trans_model.log_prob)
# obs_probs = np.exp(vbhmm.obs_model.log_likelihood(obs))

delta, predicted_sequence = vbhmm.viterbi(list(obs))


print("Predicted: \n", predicted_sequence[0])
print("True: \n", labels[0])

# plt.plot(log_liks)
# plt.show()