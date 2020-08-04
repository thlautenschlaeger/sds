from sds_bayesian_numpy.ext.utils import ensure_args_are_viable_lists
from sds_bayesian_numpy.vbhmm import VBHMM
from sds_bayesian_numpy.initial import GaussianInitState, GaussianBayesianInitState, GaussianArInitState, ArGaussianBayesianInitState
from sds_bayesian_numpy.observations import AutoRegressiveGaussianObservation, AutoRegressiveGaussianBayesianObservation, MultiAutoRegressiveGaussianBayesianObservation
import numpy as np

class VBARHMM(VBHMM):

    def __init__(self, obs, n_states, act,
                 init_prior, trans_prior, obs_prior):

        super(VBARHMM, self).__init__(obs, n_states, act,
                                      init_prior, trans_prior, obs_prior)

        self.init_obs_model = GaussianInitState(self.n_states, self.obs_dim, obs_prior)
        self.obs_model = AutoRegressiveGaussianBayesianObservation(self.n_states, self.obs_dim,
                                                                   self.act_dim, self.obs_prior)

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act):
        loginit = self.init_model.log_init
        logtrans = self.trans_model.log_transitions(obs, act)

        ilog = self.init_obs_model.log_likelihood(obs)
        arlog = self.obs_model.log_likelihood_bayes(obs, act)

        logobs = []
        for _ilog, _arlog in zip(ilog, arlog):
            logobs.append(np.vstack((_ilog, _arlog)))

        return loginit, logtrans, logobs

    def sample(self, horizon=None):
        state = []
        obs = []

        for n in range(len(horizon)):
            _obs = np.zeros((horizon[n], self.obs_dim))
            _state = np.zeros((horizon[n],), np.int64)

            _state[0] = self.init_model.sample()
            _obs[0, :] = self.init_obs_model.sample(_state[0])
            for t in range(1, horizon[n]):
                _state[t] = self.trans_model.sample(_state[t - 1], _obs[t - 1, :])

                _obs[t, :] = self.obs_model.sample(_state[t], _obs[t - 1, :])

            state.append(_state)
            obs.append(_obs)

        return state, obs