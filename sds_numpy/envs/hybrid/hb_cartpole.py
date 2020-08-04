import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def end2ang(x):
    _state = np.zeros((4, ))
    _state[0] = x[0]
    _state[1] = np.arctan2(x[2], x[1])
    _state[2] = x[3]
    _state[3] = x[4]
    return _state


class HybridCartpole(gym.Env):

    def __init__(self, rarhmm):
        self.dm_state = 4
        self.dm_act = 1
        self.dm_obs = 4

        self._dt = 0.01

        self._sigma = 1e-4

        self._global = True

        # g = [x, th, dx, dth]
        self._goal = np.array([0., 0., 0., 0.])
        self._goal_weight = - np.array([1e0, 2e0, 1e-1, 1e-1])

        # x = [x, th, dx, dth]
        self._state_max = np.array([5., np.inf, 5., 10.])

        # x = [x, th, dx, dth]
        self._obs_max = np.array([5., np.inf, 5., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max,
                                            dtype=np.float64)

        self._act_weight = - np.array([1e-2])
        self._act_max = 5.0
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,),
                                       dtype=np.float64)

        self.state = None
        self.np_random = None

        rarhmm.learn_ctl = False
        self.rarhmm = rarhmm

        self.hist_obs = np.empty((0, self.dm_obs))
        self.hist_act = np.empty((0, self.dm_act))

        self.seed()

    @property
    def xlim(self):
        return self._state_max

    @property
    def ulim(self):
        return self._act_max

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return self._goal

    def dynamics(self, xhist, uhist):
        xhist = np.atleast_2d(xhist)
        uhist = np.atleast_2d(uhist)

        # filter hidden state
        b = self.rarhmm.filter(xhist, uhist)[0][-1, ...]

        # evolve dynamics
        x, u = xhist[-1, :], uhist[-1, :]
        zn, xn = self.rarhmm.step(x, u, b, stoch=False)

        return zn, xn

    def reward(self, x, u):
        _x = end2ang(x)
        return (_x - self._goal).T @ np.diag(self._goal_weight) @ (_x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        # apply action constraints
        _act = np.clip(act, -self._act_max, self._act_max)
        self.hist_act = np.vstack((self.hist_act, _act))

        # compute reward
        rwrd = self.reward(self.obs, _act)

        # evolve dynamics
        _, self.obs = self.dynamics(self.hist_obs, self.hist_act)
        self.hist_obs = np.vstack((self.hist_obs, self.obs))

        return self.obs, rwrd, False, {}

    def reset(self):
        self.hist_obs = np.empty((0, self.dm_obs))
        self.hist_act = np.empty((0, self.dm_act))

        _state = self.rarhmm.init_state.sample()
        self.obs = self.rarhmm.init_observation.sample(z=_state)

        self.hist_obs = np.vstack((self.hist_obs, self.obs))

        return self.obs

    # following functions for plotting
    def fake_step(self, value, act):
        # switch to observation space
        _obs = value

        # apply action constraints
        _act = np.clip(act, -self._act_max, self._act_max)

        # evolve dynamics
        _nxt_state, _nxt_obs = self.dynamics(_obs, _act)

        return _nxt_state, _nxt_obs


class HybridCartpoleWithCartesianObservation(HybridCartpole):

    def __init__(self, rarhmm):
        super(HybridCartpoleWithCartesianObservation, self).__init__(rarhmm)
        self.dm_obs = 5

        # o = [x, cos, sin, xd, thd]
        self._obs_max = np.array([5., 1., 1., 5., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max,
                                            dtype=np.float64)

    def observe(self, x):
        return np.array([x[0],
                         np.cos(x[1]),
                         np.sin(x[1]),
                         x[2],
                         x[3]])


    # def fake_step(self, x, u):
