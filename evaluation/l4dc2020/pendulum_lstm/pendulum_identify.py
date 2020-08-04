import numpy as np
import gym
from sds_numpy.utils import sample_env
from evaluation.l4dc2020.pendulum_lstm.lstm import LSTM, train_lstm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score


env = gym.make('Pendulum-ID-v1')
# env = gym.make('Pendulum-v0')
env._max_episode_steps = 5000
env.unwrapped._dt = 0.01
env.unwrapped._sigma = 1e-4

nb_train_rollouts, nb_test_rollouts = 20, 15
nb_train_steps, nb_test_steps = 250, 200

train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

train_obs_in = [train_obs[i][:-1] for i in range(len(train_obs))]
train_act_in = [train_act[i][:-1] for i in range(len(train_act))]
train_in = torch.tensor(np.hstack([np.vstack(train_obs_in), np.vstack(train_act_in)]), dtype=torch.float32)[:, None]

train_obs_out = [train_obs[i][1:] for i in range(len(train_obs))]
train_act_out = [train_act[i][1:] for i in range(len(train_act))]
train_out = torch.tensor(np.vstack(train_obs_out), dtype=torch.float32)[:, None]

test_obs_in = [test_obs[i][:-1] for i in range(len(test_obs))]
test_act_in = [test_act[i][:-1] for i in range(len(test_act))]
test_in = torch.tensor(np.hstack([np.vstack(test_obs_in), np.vstack(test_act_in)]), dtype=torch.float32)[:, None]

test_obs_out = [test_obs[i][1:] for i in range(len(test_obs))]
test_act_out = [test_act[i][1:] for i in range(len(test_act))]
test_out = torch.tensor(np.vstack(test_obs_out), dtype=torch.float32)[:, None]


lstm_input_size = train_in.shape[2]
output_dim = lstm_input_size - 1
num_layers = 2
hidden_dim = 128
batch_size = 256

model = LSTM(lstm_input_size, hidden_dim, output_dim=output_dim, n_layers=num_layers)

model = train_lstm(train_in, train_out, model, batch_size=batch_size, epochs=700)
pred = []
p_steps = 5
loss = 0
trues = []
preds = []
for i in range(len(test_obs_in)):
    inp = test_in[i][None]
    h = model.init_hidden(1)
    for p, act in zip(range(p_steps), torch.tensor(test_act_in[i][:p_steps], dtype=torch.float32)):
        h = tuple([e.data for e in h])
        out, h = model(inp, h)
        pred.append(out)
        trues.append(test_obs_out[i][p])
        preds.append(out.squeeze().detach().numpy())
        loss += (np.sum(out.squeeze().detach().numpy() - test_obs_out[i][p]))**2
        inp = torch.cat([out, act[None][None]], dim=-1)

trues = np.array(trues)
preds = np.array(preds)
exp_var = explained_variance_score(y_true=trues, y_pred=preds)
loss /= len(test_obs_in)
print(loss, exp_var)

# plt.plot(np.array([np.array(p.squeeze().detach()) for p in pred]), color='green')
# plt.plot(np.array(test_out[:p_steps].squeeze()), color='orange')
# plt.show()


