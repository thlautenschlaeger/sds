import numpy as np
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal as mvn

from sds_bayesian_numpy.ext.calc import weighted_lin_reg
from sds_bayesian_numpy.ext.stats import multivariate_normal_logpdf as mvn_logpdf
from sds_bayesian_numpy.ext.stats import multivariate_studentst_logpdf as mv_studentst_logpdf
from scipy.stats import invwishart
from scipy.stats import t
from scipy.stats import wishart
from scipy import linalg

from sds_bayesian_numpy.ext.utils import timeseries_to_kernel


class GaussianBayesianObservation:
    def __init__(self, n_states, obs_dim, act_dim, prior={'W0': 1, 'nu0': 1, 'm0': 1, 'beta0': 1}, reg=1e-128):
        self.n_states = n_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.prior = prior
        self.posterior = self.init_posterior()
        self.reg = reg

        self.mean = np.random.random((self.n_states, self.obs_dim))
        self.cov = np.random.random((self.n_states, self.obs_dim, self.obs_dim))
        for k in range(self.n_states):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] = self.cov[k] + self.obs_dim * np.eye(self.obs_dim)

        # self.mu = np.random.random(size=(self.state_dim, self.obs_dim))
        # self._sqrt_cov = np.zeros(shape=(self.state_dim, self.obs_dim, self.obs_dim))
        # for k in range(self.state_dim):
        #     _cov = invwishart.rvs(self.prior['nu0'], self.prior['W0'])
        #     self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.obs_dim))


    def init_posterior(self):

        # Init pos. definit Wishart scale matrix
        W = np.random.random(size=(self.n_states, self.obs_dim, self.obs_dim))
        for k in range(self.n_states):
            W[k] = 0.5 * (W[k] + W[k].T)
            W[k] = W[k] + self.obs_dim * np.eye(self.obs_dim)

        nu = np.abs(np.random.random(size=self.n_states)) + self.obs_dim
        m = np.random.multivariate_normal(np.zeros(self.obs_dim), np.eye(self.obs_dim), size=self.n_states)
        beta = np.abs(np.random.random(size=self.n_states))

        return {'W': W, 'nu': nu, 'm': m, 'beta': beta}

    @property
    def params(self):
        return self.mean, self.cov

    @params.setter
    def params(self, value):
        self.mean, self.cov = value[0], value[1]

    def sig(self, k):
        return self.cov[k]

    def mean_prediction(self, z, x, u=None):
        _x = mvn(self.mean[z], cov=self.cov[z]).mean
        return np.atleast_1d(_x)

    def update_gauss_params(self, x, gamma):
        _norm = 0
        mu = 0
        sig = 0
        for _x, _gamma in zip(x, gamma):
            _norm += np.sum(_gamma[:, :, None], axis=0) #+ self.reg
            mu += np.sum(_gamma[:, :, None] * _x[:, None, :], axis=0)

        mu /= _norm + self.reg

        for _x, _gamma in zip(x, gamma):
            resid = _x[:, None, :] - mu
            sig += np.sum(_gamma[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)

        sig /= _norm[:, None]

        return mu, sig

    def log_likelihood(self, x, u):
        loglik = []
        for _x in x:
            _loglik = np.empty(shape=(self.n_states, len(_x)))
            for k in range(self.n_states):
                _loglik[k] = mvn.logpdf(_x, self.mean[k], self.cov[k])
            loglik.append(_loglik.T)

        return loglik


    # log lamb
    @property
    def log_lamb(self):
        loglamb = np.empty(shape=self.n_states)
        for k in range(self.posterior['nu'].shape[0]):
            _tmp = self.obs_dim * np.log(2) + np.log(np.linalg.det(self.posterior['W'][k]))
            loglamb[k] = np.sum(
                [digamma((self.posterior['nu'][k] + 1 - i) / 2)
                         for i in range(1, self.obs_dim + 1)]) + _tmp
        return loglamb

    def param_posterior_estimate(self, x, u):
        D = self.obs_dim
        post_ests = []
        for _x in x:
            _post_est = []
            for k in range(self.n_states):
                res = self.posterior['m'][k] - _x
                tmp = np.einsum('ij...,ij...->i...', np.einsum('ij...,jk...->ik...', res, self.posterior['W'][k]), res)
                tmp = D * (1 / self.posterior['beta'][k]) + self.posterior['nu'][k] * tmp
                _post_est.append(tmp)
            post_ests.append(np.vstack(_post_est))

        return post_ests

    def log_likelihood_bayes(self, x, u):
        D = self.obs_dim
        param_post_ests = self.param_posterior_estimate(x, u)

        logliks = []
        for _x, _post_est in zip(x, param_post_ests):
            _loglik = np.empty(shape=(self.n_states, _x.shape[0]))
            for k in range(self.n_states):
                _loglik[k] = 0.5 * (self.log_lamb[k] - np.log(2 * np.pi * D) - _post_est[k])
            logliks.append(_loglik.T)

        return logliks

    def m_step(self, x, gamma, u=None, **kwargs):
        self.params = self.update_gauss_params(x, gamma)

        _gamma = np.vstack(gamma)
        e_counts = _gamma.sum(axis=0)

        _x = np.vstack(x)

        self.posterior['beta'] = self.prior['beta0'] + e_counts
        self.posterior['nu'] = self.prior['nu0'] + e_counts
        self.posterior['m'] = (self.prior['beta0'] * self.prior['m0']
                               + (_gamma[:, :, None] * _x[:, None, :]).sum(axis=0)) \
                              / self.posterior['beta'][:, None]


        for k in range(self.n_states):
            _resid_0 = self.mean[k] - self.prior['m0']
            _resid_1 = _x[:, None, :] - self.mean[k]
            # _resid_2 = self.mean[k] - self.posterior['m'][k]
            NS = (_gamma[:, k, None, None] * (_resid_1[:, :, None, :] * _resid_1[:, :, :, None]).squeeze(axis=1)).sum(axis=0)

            _W_inv = linalg.inv(self.prior['W0']) + NS \
                 + ((self.prior['beta0'] * e_counts[k]) \
                    / (self.prior['beta0'] + e_counts[k])) \
                     * (_resid_0[None, :] * _resid_0[:, None])

            self.posterior['W'][k] = linalg.inv(_W_inv)


class AutoRegressiveGaussianBayesianObservation:

    def __init__(self, num_states: int, obs_dim: int, act_dim: int,
                 prior: dict={'P0':1, 'eta0':1, 'K0':1, 'M0':1}, reg=1e-128):
        self.num_states = num_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.prior = prior
        self.posterior = self.init_posterior()
        self.reg = reg


        self.mu = np.random.random((self.num_states, self.obs_dim))
        self.cov = np.random.random((self.num_states, self.obs_dim, self.obs_dim))
        for k in range(self.num_states):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] = self.cov[k] + np.eye(self.obs_dim)

    def init_posterior(self):
        P = np.random.random(size=(self.num_states, self.obs_dim, self.obs_dim))
        K = np.random.random(size=(self.num_states, self.obs_dim + self.act_dim + 1, self.obs_dim + self.act_dim + 1))
        for k in range(self.num_states):
            P[k] = 0.5 * (P[k] + P[k].T)
            P[k] = P[k] + self.obs_dim * np.eye(self.obs_dim)
            K[k] = 0.5 * (K[k] + K[k].T)
            K[k] += np.eye(self.obs_dim + self.act_dim + 1)

        eta = np.abs(np.random.random(size=self.num_states)) + self.obs_dim
        M = np.random.multivariate_normal(np.zeros(self.obs_dim + self.act_dim + 1), np.eye(self.obs_dim + self.act_dim + 1),
                                          size=(self.num_states, self.obs_dim))

        return {'P': P, 'eta': eta, 'K': K, 'M': M}

    def mean(self, z, x, u=None):
        # return self.posterior['M'][k] @ np.hstack((1, x))
        result = self.posterior['M'][z] @ np.hstack((x, u, 1)) \
            if u is not None else self.posterior['M'][z] @ np.hstack((x, 1))
        return result

    def sig(self, z):
        # return self.posterior['P'][k]
        # return np.linalg.inv(wishart.rvs(self.posterior['eta'][k], self.posterior['P'][k]))
        return linalg.inv(wishart.mode(self.posterior['eta'][z], self.posterior['P'][z]))

    def sample(self, z, x, u=None):
        _x = mvn(self.mean(z, x, u), cov=self.sig(z)).rvs()
        return np.atleast_1d(_x)

    def mean_prediction(self, z, x, u=None):
        _x = mvn(self.mean(z, x, u), cov=self.sig(z)).mean
        return np.atleast_1d(_x)

    # @property
    def log_V(self):
        logV = np.empty(shape=self.num_states)
        log_2 = np.log(2)
        D = self.obs_dim
        for k in range(self.posterior['eta'].shape[0]):
            _tmp = D * log_2 + np.log(np.linalg.det(self.posterior['P'][k]))
            logV[k] = np.sum(
                [digamma((self.posterior['eta'][k] + 1 - i) / 2)
                 for i in range(1, self.obs_dim + 1)]) + _tmp
        return logV

    def param_posterior_estimate_matrix_normal(self, x, u):
        """ result of equation 71 """
        post_ests = []
        for _x, _u in zip(x, u):
            _post_est = []
            # _x = np.hstack((_x, _u[:, :self.act_dim]))
            # _xs = np.hstack((np.ones((_x.shape[0], 1)), _x, _u[:, :self.act_dim]))[:-1, :]

            # Stack observation with control and ones for linear regression constant term
            _xs = np.hstack((_x, _u[:, :self.act_dim], np.ones((_x.shape[0], 1))))[:-1, :]
            for k in range(self.num_states):
                K_inv = linalg.inv(self.posterior['K'][k])

                # M @ x - y
                res = np.einsum('ij...,jk...->ik...', self.posterior['M'][k], _xs.T).T - _x[1:, :]

                # (M @ x - y)^T @ P @ (M @ x - y)
                tmp1 = np.einsum('ij...,ij...->i...', np.einsum('ij...,jk...->ik...', res, self.posterior['P'][k]), res)

                # trace{K^-1 @ x @ x^T}
                tmp2 = np.einsum('ij...,jk...->jik...', _xs.T, _xs)
                tmp2 = np.einsum('...ii', np.einsum('...ij,...jk->...ik', K_inv, tmp2))

                tmp = self.posterior['eta'][k] * tmp1 + tmp2

                _post_est.append(tmp)
            post_ests.append(np.vstack(_post_est))

        return post_ests

    def log_likelihood_bayes_matrix_normal(self, x, u):
        """ The estimated log likelihood for the next step prediction """
        D = self.obs_dim
        param_post_ests = self.param_posterior_estimate_matrix_normal(x, u)
        _term = np.log(2 * np.pi) * D
        _log_V = self.log_V()
        logliks = []
        for _x, _post_est in zip(x, param_post_ests):
            _loglik = np.empty(shape=(self.num_states, _x.shape[0] - 1))
            for k in range(self.num_states):
                _loglik[k] = 0.5 * (_log_V[k] - _term  - _post_est[k])
            logliks.append(_loglik.T)

        return logliks

    def log_likelihood_bayes(self, x, u):
        log_prediction = self.log_likelihood_bayes_matrix_normal(x, u)

        return log_prediction

    def m_step(self, x, gamma, u, **kwargs):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            # Stack xs for computation with constant c
            # xs.append(np.hstack((np.ones((_x.shape[0], 1)), _x, _u[:, :self.act_dim]))[:-1, :])
            xs.append(np.hstack((_x, _u[:, :self.act_dim], np.ones((_x.shape[0], 1))))[:-1, :])
            ys.append(_x[1:,])
            ws.append(_w[1:,])

        xs = np.vstack(xs)
        ys = np.vstack(ys)
        _ws = np.vstack(ws)

        _ws = _ws.T
        tmp1 = self.prior['M0'] @ self.prior['K0']
        tmp2 = linalg.inv(self.prior['P0'])
        tmp3 = self.prior['M0'] @ self.prior['K0'] @ self.prior['M0'].T
        for k in range(self.num_states):
            self.posterior['K'][k] = np.einsum('ij,jk->ik', (_ws[k][:, None] * xs).T, xs) + self.prior['K0']
            self.posterior['M'][k] = (np.einsum('ij,jk->ik', (_ws[k][:, None] * ys).T, xs) + tmp1) @ linalg.inv(
                self.posterior['K'][k])
            tmp = tmp2 + tmp3 + np.einsum('ij,jk->ik', (_ws[k][:, None] * ys).T, ys) - self.posterior['M'][k] @ self.posterior['K'][k] @ self.posterior['M'][k].T
            self.posterior['P'][k] = linalg.inv(tmp)

        e_counts = _ws.sum(axis=1)
        self.posterior['eta'] = self.prior['eta0'] + e_counts


class MultiAutoRegressiveGaussianBayesianObservation:

    def __init__(self, num_states: int, obs_dim: int,
                 prior: dict = {'P0': 1, 'eta0': 1, 'K0': 1, 'M0': 1}, reg=1e-128, ar_steps=15):
        self.num_states = num_states
        self.obs_dim = obs_dim
        self.prior = prior
        self.ar_steps = ar_steps
        self.posterior = self.init_posterior()
        self.reg = reg

        # self.A = np.random.random((self.state_dim, self.obs_dim, self.obs_dim))
        # self.c = np.random.random((self.state_dim, self.obs_dim))

        self.mu = np.random.random((self.num_states, self.obs_dim))
        self.cov = np.random.random((self.num_states, self.obs_dim, self.obs_dim))
        for k in range(self.num_states):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] = self.cov[k] + np.eye(self.obs_dim)

    def init_posterior(self):
        P = np.random.random(size=(self.num_states, self.obs_dim, self.obs_dim))
        K = np.random.random(size=(self.num_states, (self.obs_dim * self.ar_steps) + 1, (self.obs_dim * self.ar_steps) + 1))
        for k in range(self.num_states):
            P[k] = 0.5 * (P[k] + P[k].T)
            P[k] = P[k] + np.eye(self.obs_dim)
            K[k] = 0.5 * (K[k] + K[k].T)
            K[k] = K[k] + np.eye((self.obs_dim * self.ar_steps) + 1)

        eta = np.abs(np.random.random(size=self.num_states)) + 5
        M = np.random.multivariate_normal(np.zeros((self.obs_dim * self.ar_steps) + 1), np.eye((self.obs_dim * self.ar_steps) + 1),
                                          size=(self.num_states, self.obs_dim))

        return {'P': P, 'eta': eta, 'K': K, 'M': M}

    # def mean(self, k , x):
    #     # batch A and apply dot product over batches of x
    #     return np.einsum('kh,...h->...k', self.A[k,...], x) + self.c[k,:]

    def mean(self, k, x):
        return self.posterior['M'][k] @ np.hstack((1, x))

    def sig(self, k):
        return linalg.inv(wishart.rvs(self.posterior['eta'][k], self.posterior['P'][k]))

    def sample(self, k, x):
        _x = mvn(self.mean(k, x), cov=self.sig(k)).rvs()
        return np.atleast_1d(_x)

    @property
    def log_V(self):
        logV = np.empty(shape=self.num_states)
        for k in range(self.posterior['eta'].shape[0]):
            _tmp = self.obs_dim * np.log(2) + np.log(np.linalg.det(self.posterior['P'][k]))
            logV[k] = np.sum(
                [digamma((self.posterior['eta'][k] + 1 - i) / 2)
                 for i in range(1, self.obs_dim + 1)]) + _tmp
        return logV

    def param_posterior_estimate_matrix_normal(self, x):
        """ result of equation 71 """
        post_ests = []
        for _x in x:
            _post_est = []
            _xslol = np.hstack((np.ones((_x.shape[0], 1)), _x))[:-1, :]
            _xs = timeseries_to_kernel(_x, self.ar_steps)
            _xs = np.hstack((np.ones((_xs.shape[0], 1)), _xs))[:-1, :]
            # np.vstack(np.ones((_x.shape[0], 1)), timeseries_to_kernel(_x, self.ar_size))
            for k in range(self.num_states):
                K_inv = linalg.inv(self.posterior['K'][k])

                # M @ x - y
                res = np.einsum('ij...,jk...->ik...', self.posterior['M'][k], _xs.T).T - _x[self.ar_steps:, :]

                tmp1 = np.einsum('ij...,ij...->i...', np.einsum('ij...,jk...->ik...', res, self.posterior['P'][k]),
                                 res)
                tmp2 = np.einsum('ij...,jk...->jik...', _xs.T, _xs)

                # trace{K^-1 @ x @ x^T}
                tmp2 = np.einsum('...ii', np.einsum('...ij,...jk->...ik', K_inv, tmp2))

                tmp = self.posterior['eta'][k] * tmp1 + tmp2

                _post_est.append(tmp)
            post_ests.append(np.vstack(_post_est))

        return post_ests


    def log_likelihood_bayes_matrix_normal(self, x):
        """ The estimated log likelihood for the next step prediction """
        D = self.obs_dim
        param_post_ests = self.param_posterior_estimate_matrix_normal(x)
        _term = np.log(2 * np.pi * D)
        logliks = []
        for _post_est in param_post_ests:
            _loglik = np.empty(shape=(self.num_states, _post_est[0].shape[0]))
            for k in range(self.num_states):
                _loglik[k] = 0.5 * (self.log_V[k] - _term  - _post_est[k])
            logliks.append(_loglik.T)

        return logliks

    def log_likelihood_bayes(self, x):
        log_prediction = self.log_likelihood_bayes_matrix_normal(x)

        return log_prediction


    def m_step(self, x, gamma, **kwargs):
        xs, ys, ws = [], [], []
        for _x, _w in zip(x, gamma):
            # Stack xs for computation with constant c
            _xs = timeseries_to_kernel(_x, self.ar_steps)
            xs.append(np.hstack((np.ones((_xs.shape[0], 1)), _xs))[:-1, :])
            # xs.append(np.hstack((np.ones((_x.shape[0], 1)), _x))[:-1, :])
            ys.append(_x[self.ar_steps:, ])
            ws.append(_w[self.ar_steps:, ])

        xs = np.vstack(xs)
        ys = np.vstack(ys)
        _ws = np.vstack(ws)

        _ws = _ws.T
        tmp1 = self.prior['M0'] @ self.prior['K0']
        tmp2 = linalg.inv(self.prior['P0'])
        tmp3 = self.prior['M0'] @ self.prior['K0'] @ self.prior['M0'].T
        for k in range(self.num_states):
            self.posterior['K'][k] = np.einsum('ij,jk->ik', (_ws[k][:, None] * xs).T, xs) + self.prior['K0']
            self.posterior['M'][k] = (np.einsum('ij,jk->ik', (_ws[k][:, None] * xs).T, ys).T + tmp1) @ linalg.inv(self.posterior['K'][k])
            tmp = tmp2 + tmp3 + np.einsum('ij,jk->ik', (_ws[k][:, None] * ys).T, ys) - self.posterior['M'][k] @ self.posterior['K'][k] @ self.posterior['M'][k].T
            self.posterior['P'][k] = linalg.inv(tmp)

        e_counts = _ws.sum(axis=1)
        self.posterior['eta'] = self.prior['eta0'] + e_counts


class AutoRegressiveGaussianObservation:

    def __init__(self, num_states: int, obs_dim: int, prior: dict={'W0': 1, 'nu0': 1, 'm0': 1, 'beta0': 1}):
        self.num_states = num_states
        self.obs_dim = obs_dim
        self.prior = prior
        self.posterior = self.init_posterior()

        self.A = np.random.random((self.num_states, self.obs_dim, self.obs_dim))
        self.c = np.random.random((self.num_states, self.obs_dim))

        self.cov = np.random.random((self.num_states, self.obs_dim, self.obs_dim))
        for k in range(self.num_states):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] = self.cov[k] + np.eye(self.obs_dim)

    def init_posterior(self):

        # Init pos. definit Wishart scale matrix
        W = np.random.random(size=(self.num_states, self.obs_dim, self.obs_dim))
        for k in range(self.num_states):
            W[k] = 0.5 * (W[k] + W[k].T)
            W[k] = W[k] + np.eye(self.obs_dim)

        nu = np.abs(np.random.random(size=self.num_states)) + self.obs_dim
        m = np.random.multivariate_normal(np.zeros(self.obs_dim), np.eye(self.obs_dim), size=self.num_states)
        beta = np.abs(np.random.random(size=self.num_states))

        return {'W': W, 'nu': nu, 'm': m, 'beta': beta}

    @property
    def params(self):
        return self.A, self.c, self.cov

    @params.setter
    def params(self, value):
        self.A, self.c, self.cov = value

    def mean(self, k , x):
        # batch A and apply dot product over batches of x
        return np.einsum('kh,...h->...k', self.A[k,...], x) + self.c[k,:]

    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([mvn_logpdf(_x[1:, :], self.mean(k, _x[:-1, :]), self.cov[k])
                                       for k in range(self.num_states)])
            loglik.append(_loglik)

        return loglik

    def m_step(self, x, gamma, **kwargs):
        xs, ys, ws = [], [], [],
        for _x, _w in zip(x, gamma):
            # Stack xs for computation with constant c
            xs.append(np.hstack((_x[:-1, ], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:,])
            ws.append(_w[1:,])

        for k in range(self.num_states):
            self.A[k,:], self.c[k], self.cov[k] = weighted_lin_reg(np.vstack(xs), np.vstack(ys), np.vstack(ws)[:,k],
                                                      fit_intercept=True)