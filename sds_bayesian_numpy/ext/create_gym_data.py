import gym
import numpy as np

def create_trajectory(env: gym.Env, max_horizon: int):
    env.reset()

    rewards = np.empty([max_horizon, 1])
    obs = np.empty((max_horizon, env.observation_space.shape[0]))
    act_dim = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
    actions = np.empty((max_horizon, act_dim))

    for i in range(max_horizon):
        _action = env.action_space.sample()
        _obs, reward, done, info = env.step(_action)
        rewards[i] = reward
        obs[i] = _obs
        actions[i] = _action
        # env.render()
        if done:
            done = False
            # obs[i] = env.reset()
            # Make trajectory smaller if done signal received before matching horizon
            rewards = np.copy(rewards[:i + 1])
            obs = np.copy(obs[:i + 1])
            actions = np.copy(actions[:i + 1])
            env.close()
            return obs, rewards, actions

    env.close()
    return obs, rewards, actions

def create_trajectories(env: gym.Env, horizons: list):

    trajectories = []
    for h in horizons:
        _obs, _rewards, _actions = create_trajectory(env, h)
        trajectories.append(dict(obs=_obs, rewards=_rewards, actions=_actions))

    return trajectories

def create_rollouts(env: gym.Env, max_horizon: int=200, n_rollouts:int=1):
    trajectories = []
    for rollout in range(n_rollouts):
        _obs, _rewards, _actions = create_trajectory(env, max_horizon)
        trajectories.append(dict(obs=_obs, rewards=_rewards, actions=_actions))

    return trajectories