import logging
from typing import Optional, Any, Dict

import numpy as np
from stable_baselines3.common.env_util import is_wrapped

from red_gym_env import RedGymEnv


class RPCEnvWrapperWorker(RedGymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from buffer import MPIRolloutBuffer
        self.prev_rewards = np.zeros(1)

        self.buffer = MPIRolloutBuffer(buffer_size=1024, observation_space=self.observation_space,
                                       action_space=self.action_space)
        self.buffer_emit_batches = self.buffer.emit_batches
        self.buffer_reset = self.buffer.reset
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

    def get_spaces(self, *args):
        return self.observation_space, self.action_space


class RPCEnvWrapper(RPCEnvWrapperWorker):
    def __init__(self, comm, remote, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm = comm
        self.remote = remote

        for method_name in dir(super()):
            if method_name.startswith("__"):
                continue

            prop = super().__getattribute__(method_name)
            if callable(prop):
                self.__setattr__(method_name, self._rpc_wrapper(prop))

    def _rpc_wrapper(self, method):
        def _wrapper(*args, **kwargs):
            assert len(kwargs) == 0, f"kwargs {list(kwargs.keys())} were send for method {method.__name__}"
            self.comm.send((method.__name__, args), self.remote)
            logging.debug(f"Sending {method.__name__} request to {self.remote}")
            return self.comm.recv(source=self.remote)

        return _wrapper
