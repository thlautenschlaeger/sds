import numpy as np
import autograd.numpy.random as npr
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal as mvn
from sds_bayesian_numpy.ext.stats import multivariate_normal_logpdf as mvn_logpdf

class InitialState:

    def __init__(self, n_states, prior={'omega0':1}):
        self.n_states = n_states
        self.prior = prior
        self.posterior = self.init_posterior()

    def init_posterior(self):
        return {'omega': np.random.random(size=(self.n_states))}

    @property
    def log_init(self):
        """ Sub normalized log probabilities
        :returns: [num_states]
        """
        _tmp = digamma(np.sum(self.posterior['omega'], axis=0))
        return np.array([digamma(omega) - _tmp for omega in self.posterior['omega']])

    @property
    def log_prob(self):
        """ Normalized log probability
        :returns: [num_states]
        """
        return self.log_init - logsumexp(self.log_init)

    @property
    def initial_distribution(self):
        return np.exp(self.log_prob)

    def sample(self):
        return npr.choice(self.n_states, p=self.initial_distribution)

    def m_step(self, gamma, **kwargs):
        _gamma = sum([_w[0, :] for _w in gamma])
        _gamma = _gamma / sum(_gamma)

        self.posterior['omega'] = self.prior['omega0'] + _gamma


class GaussianInitState:

    def __init__(self, state_dim, obs_dim, prior=None, reg=1e-128):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reg = reg

        self.mean = np.random.random((state_dim, obs_dim))
        self.cov = np.random.random((state_dim, obs_dim, obs_dim))
        for k in range(state_dim):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] += self.obs_dim * np.eye(self.obs_dim)


    @property
    def params(self):
        return self.mean, self.cov

    @params.setter
    def params(self, value):
        self.mean, self.cov = value[0], value[1]

    def sample(self, z):
        _x = mvn(mean=self.mean[z], cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def log_likelihood(self, x, u=None):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([mvn.logpdf(_x[0, :], self.mean[k], self.cov[k], allow_singular=True)
                                       for k in range(self.state_dim)])
            loglik.append(_loglik)
        return loglik

    def update_gauss_params(self, x, gamma):
        _norm = 0
        mu = 0
        sig = 0
        for _x, _gamma in zip(x, gamma):
            _norm += np.sum(_gamma[0:, :, None], axis=0)  # + self.reg
            mu += np.sum(_gamma[0:, :, None] * _x[0:, None, :], axis=0)

        mu /= _norm + self.reg

        for _x, _gamma in zip(x, gamma):
            resid = _x[0:, None, :] - mu
            sig += np.sum(_gamma[0:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)

        sig /= _norm[:, None]

        self.params = mu, sig

    def m_step(self, x, gamma, u=None, weights=None, **kwargs):
        self.update_gauss_params(x, gamma)


class GaussianArInitState:

    def __init__(self, state_dim, obs_dim, prior=None, reg=1e-128, ar_steps=15):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reg = reg
        self.ar_steps = ar_steps

        self.mean = np.random.random((state_dim, obs_dim))
        self.cov = np.random.random((state_dim, obs_dim, obs_dim))
        for k in range(state_dim):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] += np.eye(self.obs_dim)


    @property
    def params(self):
        return self.mean, self.cov

    @params.setter
    def params(self, value):
        self.mean, self.cov = value[0], value[1]

    def sample(self, z):
        _x = mvn(mean=self.mean[z], cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([mvn.logpdf(_x[0:self.ar_steps, :], self.mean[k], self.cov[k], allow_singular=True)
                                       for k in range(self.state_dim)])
            loglik.append(_loglik)
        return loglik

    def update_gauss_params(self, x, gamma):
        _norm = 0
        mu = 0
        sig = 0
        for _x, _gamma in zip(x, gamma):
            _norm += np.sum(_gamma[0:self.ar_steps:, :, None], axis=0)  # + self.reg
            mu += np.sum(_gamma[0:self.ar_steps:, :, None] * _x[0:self.ar_steps:, None, :], axis=0)

        mu /= _norm + self.reg

        for _x, _gamma in zip(x, gamma):
            resid = _x[0:self.ar_steps:, None, :] - mu
            sig += np.sum(_gamma[0:self.ar_steps:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)

        sig /= _norm[:, None]

        self.params = mu, sig

    def m_step(self, x, gamma, weights=None, **kwargs):
        self.update_gauss_params(x, gamma)


class GaussianBayesianInitState:

    def __init__(self, state_dim, obs_dim, prior=None, reg=1e-128):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reg = reg
        self.prior = prior

        self.posterior = self.init_posterior()

        self.mean = np.random.random((state_dim, obs_dim))
        self.cov = np.random.random((state_dim, obs_dim, obs_dim))
        for k in range(state_dim):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] += np.eye(self.obs_dim)

    def init_posterior(self):

        # Init pos. definit Wishart scale matrix
        W = np.random.random(size=(self.state_dim, self.obs_dim, self.obs_dim))
        for k in range(self.state_dim):
            W[k] = 0.5 * (W[k] + W[k].T)
            W[k] = W[k] + np.eye(self.obs_dim)

        nu = np.abs(np.random.random(size=self.state_dim)) + 5 #+ self.obs_dim
        m = np.random.multivariate_normal(np.zeros(self.obs_dim), np.eye(self.obs_dim), size=self.state_dim)
        beta = np.abs(np.random.random(size=self.state_dim))

        return {'W': W, 'nu': nu, 'm': m, 'beta': beta}

    @property
    def params(self):
        return self.mean, self.cov

    @params.setter
    def params(self, value):
        self.mean, self.cov = value[0], value[1]

    # log lamb
    @property
    def log_lamb(self):
        loglamb = np.empty(shape=self.state_dim)
        for k in range(self.posterior['nu'].shape[0]):
            _tmp = self.obs_dim * np.log(2) + np.log(np.linalg.det(self.posterior['W'][k]))
            loglamb[k] = np.sum(
                [digamma((self.posterior['nu'][k] + 1 - i) / 2)
                         for i in range(1, self.obs_dim + 1)]) + _tmp
        return loglamb

    def sample(self, z):
        _x = mvn(mean=self.mean[z], cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def param_posterior_estimate(self, x):
        D = self.obs_dim
        post_ests = []
        for _x in x:
            _post_est = []
            for k in range(self.state_dim):
                res = (self.posterior['m'][k] - _x[0:])[:, None]
                tmp = res.T @ self.posterior['W'][k] @ res
                tmp = D * (1 / self.posterior['beta'][k]) + self.posterior['nu'][k] * tmp
                _post_est.append(tmp)
            post_ests.append(np.vstack(_post_est))

        return post_ests

    def log_likelihood_bayes(self, x):
        D = self.obs_dim
        param_post_ests = self.param_posterior_estimate(x)

        logliks = []
        for _x, _post_est in zip(x, param_post_ests):
            _loglik = np.empty(shape=(self.state_dim, 1))
            for k in range(self.state_dim):
                _loglik[k] = 0.5 * (self.log_lamb[k] - np.log(2 * np.pi * D) - _post_est[k])
            logliks.append(_loglik.T)

        return logliks

    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([mvn.logpdf(_x[0, :], self.mean[k], self.cov[k], allow_singular=True)
                                       for k in range(self.state_dim)])
            loglik.append(_loglik)
        return loglik

    def update_gauss_params(self, x, gamma):
        _norm = 0
        mu = 0
        sig = 0
        for _x, _gamma in zip(x, gamma):
            _norm += np.sum(_gamma[0:, :, None], axis=0)  # + self.reg
            mu += np.sum(_gamma[0:, :, None] * _x[:, None, :], axis=0)

        mu /= _norm + self.reg

        for _x, _gamma in zip(x, gamma):
            resid = _x[0:, None, :] - mu
            sig += np.sum(_gamma[0:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)

        sig /= _norm[:, None]

        self.params = mu, sig

    def m_step(self, x, gamma, weights=None, **kwargs):
        self.update_gauss_params(x, gamma)

        _gamma = np.vstack([g[0] for g in gamma])
        e_counts = _gamma.sum(axis=0)

        _x = np.vstack([_x[0] for _x in x])

        self.posterior['beta'] = self.prior['beta0'] + e_counts
        self.posterior['nu'] = self.prior['nu0'] + e_counts
        self.posterior['m'] = (self.prior['beta0'] * self.prior['m0']
                               + (_gamma[:, :, None] * _x[:, None, :]).sum(axis=0)) \
                              / self.posterior['beta'][:, None]


        for k in range(self.state_dim):
            _resid_0 = self.mean[k] - self.prior['m0']
            _resid_1 = _x[:, None, :] - self.mean[k]
            # _resid_2 = self.mean[k] - self.posterior['m'][k]
            NS = (_gamma[:, k, None, None] * (_resid_1[:, :, None, :] * _resid_1[:, :, :, None]).squeeze(axis=1)).sum(axis=0)

            _W_inv = np.linalg.inv(self.prior['W0']) + NS \
                 + ((self.prior['beta0'] * e_counts[k]) \
                    / (self.prior['beta0'] + e_counts[k])) \
                     * (_resid_0[None, :] * _resid_0[:, None])

            self.posterior['W'][k] = np.linalg.inv(_W_inv)

class ArGaussianBayesianInitState:

    def __init__(self, state_dim, obs_dim, prior=None, reg=1e-128, ar_steps=10):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reg = reg
        self.prior = prior
        self.ar_steps = ar_steps

        self.posterior = self.init_posterior()

        self.mean = np.random.random((state_dim, obs_dim))
        self.cov = np.random.random((state_dim, obs_dim, obs_dim))
        for k in range(state_dim):
            self.cov[k] = 0.5 * (self.cov[k] + self.cov[k].T)
            self.cov[k] += np.eye(self.obs_dim)

    def init_posterior(self):

        # Init pos. definit Wishart scale matrix
        W = np.random.random(size=(self.state_dim, self.obs_dim, self.obs_dim))
        for k in range(self.state_dim):
            W[k] = 0.5 * (W[k] + W[k].T)
            W[k] = W[k] + np.eye(self.obs_dim)

        nu = np.abs(np.random.random(size=self.state_dim)) + 5 #+ self.obs_dim
        m = np.random.multivariate_normal(np.zeros(self.obs_dim), np.eye(self.obs_dim), size=self.state_dim)
        beta = np.abs(np.random.random(size=self.state_dim))

        return {'W': W, 'nu': nu, 'm': m, 'beta': beta}

    @property
    def params(self):
        return self.mean, self.cov

    @params.setter
    def params(self, value):
        self.mean, self.cov = value[0], value[1]

    # log lamb
    @property
    def log_lamb(self):
        loglamb = np.empty(shape=self.state_dim)
        for k in range(self.posterior['nu'].shape[0]):
            _tmp = self.obs_dim * np.log(2) + np.log(np.linalg.det(self.posterior['W'][k]))
            loglamb[k] = np.sum(
                [digamma((self.posterior['nu'][k] + 1 - i) / 2)
                         for i in range(1, self.obs_dim + 1)]) + _tmp
        return loglamb

    def sample(self, z):
        _x = mvn(mean=self.mean[z], cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def param_posterior_estimate(self, x):
        D = self.obs_dim
        post_ests = []
        for _x in x:
            _post_est = []
            for k in range(self.state_dim):
                res = (self.posterior['m'][k] - _x[0:self.ar_steps:])[:, None]
                for i in range(self.ar_steps):
                    tmp = res[i] @ self.posterior['W'][k] @ res[i].T
                    tmp = D * (1 / self.posterior['beta'][k]) + self.posterior['nu'][k] * tmp
                    _post_est.append(tmp)
            post_ests.append(np.vstack(_post_est))

        return post_ests

    def log_likelihood_bayes(self, x):
        D = self.obs_dim
        param_post_ests = self.param_posterior_estimate(x)

        logliks = []
        for _x, _post_est in zip(x, param_post_ests):
            _loglik = np.empty(shape=(self.state_dim, 1))
            for k in range(self.state_dim):
                _loglik[k] = 0.5 * (self.log_lamb[k] - np.log(2 * np.pi * D) - _post_est[k])
            logliks.append(_loglik.T)

        return logliks

    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([mvn.logpdf(_x[0, :], self.mean[k], self.cov[k], allow_singular=True)
                                       for k in range(self.state_dim)])
            loglik.append(_loglik)
        return loglik

    def update_gauss_params(self, x, gamma):
        _norm = 0
        mu = 0
        sig = 0
        for _x, _gamma in zip(x, gamma):
            _norm += np.sum(_gamma[0:self.ar_steps:, :, None], axis=0)  # + self.reg
            mu += np.sum(_gamma[0:self.ar_steps:, :, None] * _x[0:self.ar_steps:, None, :], axis=0)

        mu /= _norm + self.reg

        for _x, _gamma in zip(x, gamma):
            resid = _x[0:self.ar_steps:, None, :] - mu
            sig += np.sum(_gamma[0:self.ar_steps:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)

        sig /= _norm[:, None]

        self.params = mu, sig

    def m_step(self, x, gamma, weights=None, **kwargs):
        self.update_gauss_params(x, gamma)

        _gamma = np.vstack([g[0:self.ar_steps] for g in gamma])
        e_counts = _gamma.sum(axis=0)

        _x = np.vstack([_x[0:self.ar_steps] for _x in x])

        self.posterior['beta'] = self.prior['beta0'] + e_counts
        self.posterior['nu'] = self.prior['nu0'] + e_counts
        self.posterior['m'] = (self.prior['beta0'] * self.prior['m0']
                               + (_gamma[:, :, None] * _x[:, None, :]).sum(axis=0)) \
                              / self.posterior['beta'][:, None]


        for k in range(self.state_dim):
            _resid_0 = self.mean[k] - self.prior['m0']
            _resid_1 = _x[:, None, :] - self.mean[k]
            # _resid_2 = self.mean[k] - self.posterior['m'][k]
            NS = (_gamma[:, k, None, None] * (_resid_1[:, :, None, :] * _resid_1[:, :, :, None]).squeeze(axis=1)).sum(axis=0)

            _W_inv = np.linalg.inv(self.prior['W0']) + NS \
                 + ((self.prior['beta0'] * e_counts[k]) \
                    / (self.prior['beta0'] + e_counts[k])) \
                     * (_resid_0[None, :] * _resid_0[:, None])

            self.posterior['W'][k] = np.linalg.inv(_W_inv)

