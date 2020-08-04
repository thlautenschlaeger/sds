from sds_bayesian_numpy import initial, transitions, observations
from scipy.special import logsumexp
from sds_bayesian_numpy.ext.calc import logdotexp
import numpy as np
import autograd.numpy.random as npr
import logging
from tqdm import trange
from autograd.tracer import getval

log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)

from vbhmm.hmm.cython.vbhmm_cy import forward_cy
from vbhmm.hmm.cython.vbhmm_cy import backward_cy

from sds_bayesian_numpy.ext.utils import timeseries_to_kernel, ensure_args_are_viable_lists

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)


class VBHMM:

    def __init__(self, obs: list, n_states: int, act: list={}, init_prior: dict={}, trans_prior: dict={}, obs_prior: dict={}):
        self.obs = obs
        self.act = act
        self.n_states = n_states
        self.obs_dim = obs[0].shape[1]
        self.act_dim = 0 if len(act) == 0 else act[0].shape[1]

        # if len(act) == 0:
        #     act = []
        #     for _obs in obs:
        #         act.append(np.zeros((_obs.shape[0], self.n_states)))
        # self.act = act

        self.init_prior = init_prior
        self.trans_prior = trans_prior
        self.obs_prior = obs_prior

        self.init_model = initial.InitialState(n_states, self.init_prior)
        self.trans_model = transitions.BayesianStationaryTransition(n_states, self.obs_dim, self.act_dim, self.trans_prior)
        self.obs_model = observations.GaussianBayesianObservation(n_states, self.obs_dim, self.act_dim, self.obs_prior)

    def log_norm_local(self, u):
        norm = logsumexp(u)
        alpha = u - norm

        return alpha, norm

    def log_norm(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        _, norm = self.forward(*loglikhds)
        return np.sum(np.hstack(norm))

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act):
        return self.init_model.log_init, self.trans_model.log_transitions(obs, act), \
               self.obs_model.log_likelihood_bayes(obs, act)

    def forward(self, loginit, logtrans, logobs, logctl=None, cython=True):
        alpha, norm = [], []
        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.n_states)))

        for _logobs, _logctl, _logtrans in zip(logobs, logctl, logtrans):
            T = _logobs.shape[0]
            _alpha = np.empty((T, self.n_states))
            _norm = np.empty(T)
            if cython:
                forward_cy(to_c(loginit), to_c(_logtrans),
                           to_c(_logobs), to_c(_logctl),
                           to_c(_alpha), to_c(_norm))
            else:
                _alpha[0], _norm[0] = self.log_norm_local(loginit + _logobs[0, :] + _logctl[0, :])
                for t in range(1, T):
                    _alpha[t], _norm[t] = self.log_norm_local(_logobs[t, :] + _logctl[t, :] + logdotexp(_logtrans[t - 1].T, _alpha[t - 1]))

            alpha.append(_alpha)
            norm.append(_norm)

        return alpha, norm

    def backward(self, loginit, logtrans, logobs, scale, logctl=None, cython=True):
        beta = []

        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.n_states)))

        for _logobs, _logctl, _scale, _logtrans in zip (logobs, logctl, scale, logtrans):
            T = _logobs.shape[0]
            _beta = np.zeros((T, self.n_states))
            _tmp = np.zeros(self.n_states)
            _beta[-1] = 0

            if cython:
                backward_cy(to_c(loginit), to_c(_logtrans),
                            to_c(_logobs), to_c(_logctl),
                            to_c(_beta), to_c(_scale))
            else:
                for t in reversed(range(0, T - 1)):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            _tmp[j] = _logtrans[t, i, j] + _logobs[t + 1, j] + _logctl[t + 1, j] + _beta[t + 1, j]
                        _beta[t, i] = logsumexp(_tmp) - _scale[t + 1]

            beta.append(_beta)

        return beta

    def posterior(self, alpha, beta):
        gamma = []
        for _alpha, _beta in zip(alpha, beta):
            T = _alpha.shape[0]
            _gamma = np.empty(_alpha.shape)
            for t in range(T):
                _norm = logsumexp(_alpha[t] + _beta[t])
                for j in range(self.n_states):
                    _gamma[t, j] = _alpha[t, j] + _beta[t, j] - _norm
            gamma.append(np.exp(_gamma))

        return gamma

    def joint_posterior(self, alpha, beta, loginit, logtrans, logobs, logctl=None):
        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.n_states)))

        xi = []
        for _alpha, _beta, _logobs, _logctl, _logtrans in zip(alpha, beta, logobs, logctl, logtrans):
            _xi = _alpha[:-1, :, None] + _beta[1:, None, :] + _logtrans \
                 + _logobs[1:, :][:, None, :] + _logctl[1:, :][:, None, :]

            _xi -= _xi.max((1, 2))[:, None, None]
            _xi = np.exp(_xi)
            _xi /= _xi.sum((1, 2))[:, None, None]

            xi.append(_xi)

        return xi

    def e_step(self, obs, act):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)
        xi = self.joint_posterior(alpha, beta, *loglikhds)

        return gamma, xi, np.hstack(norm).sum(axis=0)

    def m_step(self, gamma, xi, obs, act, init_mstep_kwargs,
               trans_mstep_kwargs, obs_mstep_kwargs, **kwargs):

        if hasattr(self, 'init_obs_model'):
            self.init_obs_model.m_step(obs, gamma)

        self.init_model.m_step(gamma, **init_mstep_kwargs)
        self.trans_model.m_step(xi, obs, act, **trans_mstep_kwargs)
        self.obs_model.m_step(obs, gamma, act, **obs_mstep_kwargs)

    @ensure_args_are_viable_lists
    def em(self, obs, act=None, n_iter=100, prec=10e-4, init_mstep_kwargs={},
           trans_mstep_kwargs={}, obs_mstep_kwargs={}, **kwargs):
        process_id = kwargs.get('process_id', 0)

        log_liks = []
        log_liks.append(self.log_norm(obs, act))

        _delta = - np.Inf
        _count = 0
        log_lik = _delta
        pbar = trange(n_iter, position=process_id)
        pbar.set_description("#{}, ll: {:.5f}".format(process_id, log_liks[-1]))
        for _ in pbar:
            prev_log_lik = log_lik
            gamma, xi, log_lik = self.e_step(obs, act)
            self.m_step(gamma, xi, obs, act, init_mstep_kwargs,
                        trans_mstep_kwargs,
                        obs_mstep_kwargs,
                        **kwargs)
            log_liks.append(log_lik)

            if _delta > 0:
                d = 'DOWN'
            else:
                d = 'UP'

            _delta = prev_log_lik - log_lik
            _count += 1
            # if hasattr(self, 'trans_type'):
            #     print("it={} ll={}  | log net weight prior: {} | log net weight posterior: {} ".format(_count, log_lik,
            #                                                                      self.trans_model.regressor.log_net_prior(),
            #                                                                      self.trans_model.regressor.log_net_post()))
            # else:
            # print("it={} ll={} | {}".format(_count, log_lik, d))
            pbar.set_description("#{}, ll: {:.5f}".format(process_id, log_liks[-1]))

        return log_liks

    def viterbi(self, obs, act=None):
        loginit, logtrans, logobs = self.log_likelihoods(obs, act)

        delta = []
        z = []
        for _logobs, _logtrans in zip(logobs, logtrans):
            T = _logobs.shape[0]

            _delta = np.zeros((T, self.n_states))
            _args = np.zeros((T, self.n_states), np.int64)
            _z = np.zeros((T, ), np.int64)

            for t in range(T - 2, -1, -1):
                _aux = _logtrans[t,:] + _delta[t + 1, :] + _logobs[t + 1, :]
                _delta[t, :] = np.max(_aux, axis=1)
                _args[t + 1, :] = np.argmax(_aux, axis=1)

            _z[0] = np.argmax(loginit + _delta[0, :] + _logobs[0, :], axis=0)
            for t in range(1, T):
                _z[t] = _args[t, _z[t - 1]]

            delta.append(_delta)
            z.append(_z)

        return delta, z

    def forecast(self, hist_obs=None, hist_act=None, nxt_act=None, horizon=None, stoch=False, average=False, stoch_reps=1):
        nxt_state = []
        nxt_obs = []
        nxt_cov = []
        nxt_var = []

        for n in range(len(horizon)):
            _hist_obs = hist_obs[n]
            _hist_act = hist_act[n]

            _nxt_act = np.zeros((horizon[n], self.act_dim)) if nxt_act is None else nxt_act[n]
            _nxt_obs = np.zeros((horizon[n] + 1, self.obs_dim))
            _nxt_cov = np.zeros((horizon[n] + 1, self.obs_dim, self.obs_dim))
            _nxt_var = np.zeros((horizon[n] + 1, self.obs_dim))
            _nxt_state = np.zeros((horizon[n] + 1,), np.int64)

            _belief = self.filter(_hist_obs, _hist_act)[0][-1, ...]

            if stoch:
                _obs_runs = _nxt_obs
                for i in range(stoch_reps):
                    _nxt_state[0] = npr.choice(self.n_states, p=_belief)
                    _nxt_obs[0, :] = _hist_obs[-1, ...]
                    for t in range(horizon[n]):
                        _nxt_state[t + 1] = self.trans_model.sample(_nxt_state[t], _nxt_obs[t, :], _nxt_act[t,:])
                        _nxt_obs[t + 1, :] = self.obs_model.sample(_nxt_state[t + 1], _nxt_obs[t, :], _nxt_act[t,:])
                        _obs_runs[t + 1, :] += _nxt_obs[t + 1, :]
                _nxt_obs = _obs_runs/stoch_reps
                _nxt_obs[0, :] = _hist_obs[-1, ...]
            else:
                if average:
                    # return empty discrete state when mixing
                    _nxt_state = None
                    # _nxt_state[0] = np.argmax(_belief)
                    _nxt_obs[0, :] = _hist_obs[-1, ...]
                    for t in range(horizon[n]):

                        # average over transitions and belief space
                        _logtrans = np.squeeze(self.trans_model.log_transition(_nxt_obs[t, :], _nxt_act[t,:])[0])
                        _trans = np.exp(_logtrans - logsumexp(_logtrans, axis=1, keepdims=True))

                        # update belief
                        _zeta = _trans.T @ _belief
                        _belief = _zeta / _zeta.sum()

                        # average observations
                        for k in range(self.n_states):
                            _nxt_obs[t + 1, :] += _belief[k] * self.obs_model.mean_prediction(k, _nxt_obs[t, :], _nxt_act[t,:])
                else:
                    _nxt_state[0] = np.argmax(_belief)
                    _nxt_obs[0, :] = _hist_obs[-1, ...]

                    for t in range(horizon[n]):
                        _nxt_state[t + 1] = self.trans_model.likeliest(_nxt_state[t], _nxt_obs[t, :], _nxt_act[t,:])
                        _nxt_obs[t + 1, :] = self.obs_model.mean_prediction(_nxt_state[t + 1], _nxt_obs[t, :], _nxt_act[t,:])
                        _nxt_cov[t + 1, :] = self.obs_model.sig(_nxt_state[t + 1])

                        _nxt_var[t + 1, :] = np.diag(_nxt_cov[t + 1])

            nxt_state.append(_nxt_state)
            nxt_obs.append(_nxt_obs)
            nxt_cov.append(_nxt_cov)
            nxt_var.append(_nxt_var)

        return nxt_state, nxt_obs, nxt_cov, nxt_var

    @ensure_args_are_viable_lists
    def filter(self, obs, act=None):
        logliklhds = self.log_likelihoods(obs, act)
        alpha, _ = self.forward(*logliklhds)
        belief = [np.exp(_alpha - logsumexp(_alpha, axis=1, keepdims=True)) for _alpha in alpha]
        return belief

    @ensure_args_are_viable_lists
    def kstep_mse(self, obs, act, horizon=1, stoch=False, average=False):

        from sklearn.metrics import mean_squared_error, \
            explained_variance_score, r2_score

        mse, smse, evar = [], [], []
        for _obs, _act in zip(obs, act):
            _hist_obs, _hist_act, _nxt_act = [], [], []
            _target, _prediction = [], []

            _nb_steps = _obs.shape[0] - horizon
            for t in range(_nb_steps):
                _hist_obs.append(_obs[:t + 1, :])
                _hist_act.append(_act[:t + 1, :])
                _nxt_act.append(_act[t: t + horizon, :])

            _hr = [horizon for _ in range(_nb_steps)]
            _, _forcast, _, _ = self.forecast(hist_obs=_hist_obs, hist_act=_hist_act,
                                       nxt_act=_nxt_act, horizon=_hr, stoch=stoch,
                                       average=average)

            for t in range(_nb_steps):
                _target.append(_obs[t + horizon, :])
                _prediction.append(_forcast[t][-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            _smse = 1. - r2_score(_target, _prediction, multioutput='variance_weighted')
            _evar = explained_variance_score(_target, _prediction, multioutput='variance_weighted')

            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)
