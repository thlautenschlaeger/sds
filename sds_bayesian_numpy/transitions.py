import numpy as np
import autograd.numpy.random as npr
from scipy.special import digamma, logsumexp
from sds_bayesian_numpy.ext.utils import adam

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.optim import Adam

from torch.utils.data import DataLoader, TensorDataset

from sds_bayesian_numpy.bayesian_nn.bnn import BNNLayer, BNNLayerLocalRep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()

class BayesianStationaryTransition:

    def __init__(self, n_states, obs_dim, act_dim, prior={'omega0': np.ones(3)}):
        """
        :param prior: contains concentration parameter for dirichlet prior
        """
        self.n_states = n_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.prior = prior
        self.posterior = self.init_posterior()

    def init_posterior(self):
        return {'omega': np.random.random(size=(self.n_states, self.n_states))}

    def sample(self, z, x=None, u=None):
        return npr.choice(self.n_states, p=self.transition_matrix[z, :])

    def likeliest(self, z, x=None, u=None):
        return np.argmax(self.transition_matrix[z, :])

    def log_prior(self):
        lp = 0
        for k in range(self.n_states):
            pass

    @property
    def log_transition(self):
        """ Sub normalized transition obtained computing the Dirichlet estimate
        :returns: [state_dim x state_dim]
        """
        logtrans = np.empty(shape=(self.n_states, self.n_states))
        for k in range(self.posterior['omega'].shape[0]):
            _tmp = digamma(np.sum(self.posterior['omega'][k], axis=0))
            logtrans[k]= [digamma(omega) - _tmp for omega in self.posterior['omega'][k]]

        return logtrans

    def log_transitions(self, x, u=None):
        """ Sub normalized transition obtained computing the Dirichlet estimate
        :returns: [state_dim x state_dim]
        """
        logtrans = np.empty(shape=(self.n_states, self.n_states))
        for k in range(self.posterior['omega'].shape[0]):
            _tmp = digamma(np.sum(self.posterior['omega'][k], axis=0))
            logtrans[k]= [digamma(omega) - _tmp for omega in self.posterior['omega'][k]]


        logtrans = [np.repeat(logtrans[None, :], _x.shape[0]-1, axis=0) for _x in x ]
        return logtrans

    @property
    def log_prob(self):
        """ Normalized transition responsibility from z_t-1 to z_t
        :returns: [state_dim x state_dim]
        """
        return (self.log_transition.T - logsumexp(self.log_transition, axis=1)).T

    @property
    def transition_matrix(self):
        """ Transition matrix with correct probabilities """
        return np.exp(self.log_prob)

    def m_step(self, xi, x, u=None):
        _counts = sum([np.sum(_xi, axis=0) for _xi in xi])
        for k in range(self.n_states):
            self.posterior['omega'][k] = self.prior['omega0'][k] + _counts[k]


class BayesianNeuralRecurrentTransition:

    def __init__(self, n_states, obs_dim, act_dim, prior=None, norm=None,
                 hidden_neurons =(12,), nonlin='tanh'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_states = n_states

        self.prior = prior

        if norm is None:
            self.norm = {'mean': np.zeros((1, self.obs_dim + self.act_dim)),
                         'std': np.ones((1, self.obs_dim + self.act_dim))}
        else:
            self.norm = norm

        self.activation = nonlin

        sizes = [self.obs_dim + self.act_dim] + list(hidden_neurons) + [self.n_states]
        self.regressor = BayesianNeuralRecurrentRegressor(sizes, prior=prior, norm=self.norm,
                                                          nonlin=nonlin)
        self.regressor.to(device)


    @property
    def params(self):
        return super(BayesianNeuralRecurrentTransition, self).params + (self.weights, self.biases)

    @property
    def transition_matrix(self):
        return to_npy(self.regressor.logmat.data)

    def log_prior(self):
        self.regressor.eval()
        if self.prior:
            return to_npy(self.regressor.log_prior())
        else:
            return self.regressor.log_prior()

    def sample(self, z, x, u=None):
        return self.maximum(z, x)

    def maximum(self, z, x, u=None):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def likeliest(self, z, x, u=None):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def log_transition(self, x, u=None):
        """ log transition for single x and u """
        self.regressor.eval()

        logtrans = []
        _logtrans = to_npy(self.regressor.forward(to_torch(np.hstack((x,u))[None])))
        logtrans.append(_logtrans)
        return logtrans

    def log_transitions(self, x, u=None):
        """ log transitions for stack of x and u """
        self.regressor.eval()

        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.act_dim]))
            _logtrans = to_npy(self.regressor.forward(to_torch(_in)))
            logtrans.append(_logtrans)
        return logtrans

    def m_step(self, xi, x, u, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.act_dim])))

        self.regressor.fit(to_torch(np.vstack(xi)), to_torch(np.vstack(xu)), **kwargs)


class BayesianNeuralRecurrentRegressor(nn.Module):
    def __init__(self, sizes, norm, prior=None, noise_tol = 0.1, rho_prior=1., nonlin='relu', lr=1e-2, use_local_rep=True):
        super(BayesianNeuralRecurrentRegressor, self).__init__()

        self.device = device
        self.sizes = sizes
        self.prior = prior
        self.n_states = sizes[-1]
        self.norm = norm
        self.use_local_rep = use_local_rep

        nlist = dict(relu=torch.relu, tanh=torch.tanh,
                     sigmoid=torch.sigmoid, softplus=F.softplus)

        self.nonlin = nlist[nonlin]
        if self.use_local_rep:
            self.layer_1 = BNNLayerLocalRep(sizes[0], sizes[1])
            # self.layer_2 = BNNLayerLocalRep(sizes[1], sizes[2])
            self.output = BNNLayerLocalRep(sizes[1], sizes[2])
        else:
            self.layer_1 = BNNLayer(sizes[0], sizes[1])
            # self.layer_2 = BNNLayer(sizes[1], sizes[2])
            self.output = BNNLayer(sizes[1], sizes[2])

        self.noise_tol = noise_tol
        self.rho_prior = rho_prior

        _mat = torch.ones(self.n_states, self.n_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat), requires_grad=True)

        self._mean = torch.as_tensor(self.norm['mean'], dtype=torch.float32, device=device)
        self._std = torch.as_tensor(self.norm['std'], dtype=torch.float32, device=device)

        if self.prior:
            if 'alpha' in self.prior and 'kappa' in self.prior:
                self._concentration = torch.zeros(self.n_states, self.n_states, dtype=torch.float32)
                for k in range(self.n_states):
                    self._concentration[k, ...] = self.prior['alpha'] * torch.ones(self.n_states)\
                            + self.prior['kappa'] * torch.as_tensor(torch.arange(self.n_states) == k, dtype=torch.float32)
                self._dirichlet = dist.dirichlet.Dirichlet(self._concentration.to(self.device))

        if self.prior and 'l2_penalty' in self.prior:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=self.prior['l2_penalty'])
        else:
            self.optim = Adam(self.parameters(), lr=lr)



    def log_prior(self):
        lp = torch.as_tensor(0., device=self.device)
        if self.prior:
            if hasattr(self, '_dirichlet'):
                _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
                lp += self._dirichlet.log_prob(_matrix.to(self.device)).sum()
        return lp

    def forward(self, xu):
        norm_xu = (xu - self._mean) / self._std
        if self.training and self.use_local_rep:
            tkl = 0
            x, klqw = self.layer_1(norm_xu)
            tkl += klqw
            # x, klqw = self.layer_2(self.nonlin(x))
            # tkl += klqw
            out, klqw = self.output(self.nonlin(x))
            tkl += klqw
            _logtrans = self.logmat[None, :, :] + out[:, None, :]
            _logtrans = _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)
            return _logtrans, tkl
        else:
            # out = self.output(self.nonlin(self.layer_2(self.nonlin(self.layer_1(norm_xu)))))
            out = self.output(self.nonlin(self.layer_1(norm_xu)))
            _logtrans = self.logmat[None, :, :] + out[:, None, :]
            return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def hmm_elbo(self, xi, xu, n_samples=1):
        logtrans = torch.empty(n_samples, xu.shape[0], self.n_states, self.n_states)
        mklq = torch.zeros(n_samples)
        if self.use_local_rep:
            for i in range(n_samples):
                logtrans[i], mklq[i] = self.forward(xu)
        else:
            for i in range(n_samples):
                logtrans[i] = self.forward(xu)
        logtrans = logtrans.mean(dim=0)
        mklq = mklq.mean()

        return torch.sum(xi * logtrans) + self.log_prior() - mklq

    def log_net_prior(self):
        return self.layer_1.log_prior + self.layer_2.log_prior + self.output.log_prior

    def log_net_post(self):
        return self.layer_1.log_post + self.layer_2.log_post + self.output.log_post

    def fit(self, xi, xu, n_iter=50, batch_size=None):
        self.train()
        data_size = xu.shape[0]
        batch_size = data_size if batch_size is None else batch_size

        dataset = TensorDataset(xi, xu)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for n in range(n_iter):
            for _xi_batch, _xu_batch in loader:
                self.optim.zero_grad()
                loss = -self.hmm_elbo(_xi_batch, _xu_batch)
                loss.backward()
                self.optim.step()