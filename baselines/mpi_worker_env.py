from typing import Optional, Any, Dict

import mpi4py.MPI
import numpy as np
from stable_baselines3.common.env_util import is_wrapped

from red_gym_env import RedGymEnv


class RPCEnvWrapperWorker(RedGymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from buffer import MPIRolloutBuffer
        self.prev_rewards = np.zeros(1)
        self.buffer = MPIRolloutBuffer(buffer_size=self.buffer_size, observation_space=self.observation_space,
                                       action_space=self.action_space)
        self.reset_info: Optional[Dict[str, Any]] = {}

    def step(self, *args):
        observation, reward, terminated, truncated, info = super().step(*args)
        # convert to SB3 VecEnv api
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation, self.reset_info = self.reset()
        return observation, reward, done, info, self.reset_info

    def buffer_compute_returns_and_advantage(self, *args, **kwargs):
        self.buffer.compute_returns_and_advantage(*args, **kwargs)
        return self.buffer.returns, self.buffer.values

    def buffer_reset(self):
        self.buffer.reset()

    def buffer_emit_batches(self):
        self.buffer.emit_batches(mpi4py.MPI.COMM_WORLD)

    def buffer_add(self, observations, actions, rewards, episode_starts, values, log_probs):
        self.buffer.add(observations, actions, self.prev_rewards, episode_starts, values, log_probs)
        self.prev_rewards = rewards

    def env_method(self, method, *args, **kwargs):
        method = getattr(self, method)
        return method(*args, **kwargs)

    def get_attr(self, prop):
        return getattr(self, prop)

    def set_attr(self, prop, value):
        return setattr(self, prop, value)

    def is_wrapped(self, data):
        return is_wrapped(self, data)

    def get_spaces(self):
        return self.observation_space, self.action_space


