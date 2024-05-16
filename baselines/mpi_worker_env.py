from typing import Optional, Any, Dict

import mpi4py.MPI
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from fast_subproc_vec_env import StaggeredSubprocVecEnv


class RPCEnvWrapperWorker(StaggeredSubprocVecEnv):
    def __init__(self, *args, gamma=0.95, gae_lambda=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        from buffer import MPIRolloutBuffer

        self.buffer_size = self.get_attr("buffer_size", indices=0)[0]
        self.prev_rewards = np.zeros(self.num_envs)
        self.buffer = MPIRolloutBuffer(
            buffer_size=self.buffer_size, observation_space=self.observation_space,
            action_space=self.action_space, n_workers=self.num_envs, gamma=gamma, gae_lambda=gae_lambda
        )
        self.reset_info: Optional[Dict[str, Any]] = {}

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

    def is_wrapped(self, data):
        return self.env_is_wrapped(data)

    def get_spaces(self):
        return self.observation_space, self.action_space
