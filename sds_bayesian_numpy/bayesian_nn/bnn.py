"""
https://arxiv.org/abs/1505.05424
Weight Uncertainty in Neural Networks - extended to continuous data
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()

def kld_cost(mu_p, sig_p, mu_q, sig_q):
    """ https://arxiv.org/abs/1312.6114 """
    # return 0.5 * (torch.log(sig_p / sig_q).pow(2) + 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()

class BNNLayer(nn.Module):
    def __init__(self, n_input, n_output, rho_prior=1.0):
        super(BNNLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.rho_prior = rho_prior

        self.W_mu = nn.Parameter(torch.Tensor(self.n_output, self.n_input).normal_(0, 0.01))
        self.W_logrho = nn.Parameter(torch.Tensor(self.n_output, self.n_input).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(self.n_output).uniform_(-0.01, 0.01))
        self.b_logrho = nn.Parameter(torch.Tensor(self.n_output).uniform_(-0.01, 0.01))

        self.W_prior = None
        self.b_prior = None
        self.W_post = None
        self.b_post = None
        self.log_post = None

        self.prior = Normal(0, self.rho_prior) # maybe change with laplace prior

    def forward(self, x):
        W_epsilon = Normal(0, self.rho_prior).sample(self.W_mu.shape)
        b_epsilon = Normal(0, self.rho_prior).sample(self.b_mu.shape)
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logrho)) * W_epsilon
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logrho)) * b_epsilon
        if not self.training:
            return F.linear(x, W, b)

        W_log_prior = self.prior.log_prob(W)
        b_log_prior = self.prior.log_prob(b)
        self.log_prior = torch.sum(W_log_prior) + torch.sum(b_log_prior)

        self.W_post = Normal(self.W_mu, torch.log(1 + torch.exp(self.W_logrho)))
        self.b_post = Normal(self.b_mu, torch.log(1 + torch.exp(self.b_logrho)))
        self.log_post = self.W_post.log_prob(W).sum() + self.b_post.log_prob(b).sum()

        return F.linear(x, W, b)

class BNNLayerLocalRep(nn.Module):
    def __init__(self, n_input, n_output, rho_prior=1.0):
        super(BNNLayerLocalRep, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.rho_prior = rho_prior

        self.W_mu = nn.Parameter(torch.Tensor(self.n_output, self.n_input).normal_(0, 0.01))
        self.W_logrho = nn.Parameter(torch.Tensor(self.n_output, self.n_input).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(self.n_output).uniform_(-0.01, 0.01))
        self.b_logrho = nn.Parameter(torch.Tensor(self.n_output).uniform_(-0.01, 0.01))

        self.W_prior = None
        self.b_prior = None
        self.W_post = None
        self.b_post = None
        self.log_post = None

        self.prior = Normal(0, self.rho_prior) # maybe change with laplace prior


    def forward(self, x):
        w_std = torch.log(1 + torch.exp(self.W_logrho))
        b_std = torch.log(1 + torch.exp(self.b_logrho))

        act_W_mu = F.linear(x, self.W_mu)
        act_W_std = torch.sqrt(F.linear(x.pow(2), w_std.pow(2)))

        if not self.training:
            return F.linear(x, self.W_mu, self.b_mu)

        W_epsilon = Normal(0, self.rho_prior).sample(act_W_mu.shape)
        b_epsilon = Normal(0, self.rho_prior).sample(b_std.shape)

        act_W_out = act_W_mu + act_W_std * W_epsilon
        act_b_out = self.b_mu + b_std * b_epsilon

        out = act_W_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1)

        kld = kld_cost(mu_p=0, sig_p=self.rho_prior, mu_q=self.W_mu, sig_q=w_std) + \
              kld_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=b_std)

        return out, kld

class BNNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_neurons, noise_tol = 0.1, rho_prior=1., nonlinear='tanh',
                 use_local_rep=True):
        super(BNNet, self).__init__()

        # Check if using local reparameterization trick
        if use_local_rep:
            self.hidden = BNNLayerLocalRep(n_input, n_hidden_neurons, rho_prior)
            self.out = BNNLayerLocalRep(n_hidden_neurons, n_output, rho_prior)
            self.use_anal_kl = True
        else:
            self.hidden = BNNLayer(n_input, n_hidden_neurons, rho_prior)
            self.out = BNNLayer(n_hidden_neurons, n_output, rho_prior)
            self.use_anal_kl = False
        self.noise_tol = noise_tol
        self.rho_prior = rho_prior

        # List of non-linearyties
        nlist = dict(relu=torch.relu, tanh=torch.tanh,
                     softmax=torch.log_softmax)
        self.nonlinearity = nlist[nonlinear]

    def forward(self, x):
        if self.use_anal_kl and self.training:
            tlqw = 0 # Total loss for approx. distribution
            x, lqw = self.hidden(x)
            tlqw += lqw
            x = self.nonlinearity(x)
            x, lqw = self.out(x)
            tlqw += lqw

            return x, tlqw

        else:
            x = self.nonlinearity(self.hidden(x))
            return self.out(x)

    def log_prior(self):
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, x, y, n_samples):
        self.train()
        outputs = torch.empty(n_samples, y.shape[0])
        log_priors = torch.empty(n_samples)
        log_posts = torch.empty(n_samples)
        log_liks = torch.empty(n_samples)

        for i in range(n_samples):
            outputs[i] = self.forward(x).reshape(-1)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_liks[i] = Normal(outputs[i], self.noise_tol).log_prob(y.reshape(-1)).sum()

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_lik = log_liks.mean()

        loss = log_post - log_prior - log_lik

        return loss


    def sample_elbo_anal_kl(self, x, y, n_samples):
        """ Samples elbo and optimizes analytically derived elbo """
        self.train()
        klq = torch.empty(n_samples)
        ml_loss = torch.empty(n_samples)
        for i in range(n_samples):
            out, klq[i]  = self.forward(x)
            ml_loss[i]  = F.mse_loss(out.reshape(-1), y.reshape(-1), reduction='sum')

        loss = klq.mean() + ml_loss.mean()

        return loss


def train_bnn(inputs, targets, model, batch_size=None, epochs=100, lr=0.1, n_samples=1, shuffle=False):
    model.train(True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = inputs.shape[0] if batch_size is None else batch_size

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if model.use_anal_kl:
        sample_elbo = model.sample_elbo_anal_kl
    else:
        sample_elbo = model.sample_elbo

    pbar = tqdm(range(epochs))

    losses = []
    for epoch in pbar:
        epoch_loss = 0
        for x, y in loader:
            loss = sample_elbo(x, y, n_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            pbar.set_description("[batch size: {} | #elbo samples: {} | epoch loss: {} ] | progress: ".format(batch_size, n_samples, epoch_loss))

        losses.append(epoch_loss)

    plt.plot(losses)
    # plt.show()

    return model