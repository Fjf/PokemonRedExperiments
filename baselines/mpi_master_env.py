import logging
import logging
import traceback
import warnings
from functools import partial
from typing import Sequence, Optional, List, Any, Type

import gymnasium as gym
import joblib
import numpy as np
import torch
from gymnasium import Env
from mpi4py import MPI
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs, VecEnvIndices
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from mpi_worker_env import RPCEnvWrapperWorker


class RPCEnvWrapper(RPCEnvWrapperWorker):
    def __init__(self, comm, remote, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm = comm
        self.remote = remote

        for method_name in dir(self):
            if method_name.startswith("_"):
                continue

            prop = getattr(self, method_name)
            if callable(prop):
                self.__setattr__(method_name, self._rpc_wrapper(prop, delayed=True))

    def _rpc_wrapper(self, method, delayed=False):
        def converter(args):
            """
            Tensors need to be moved to the CPU when sending them over MPI, as we cannot guarantee that the
             recipient has access to a GPU.
            """
            def t_c(arg):
                if type(arg) == torch.Tensor:
                    return arg.to("cpu")
                return arg

            if type(args) == tuple:
                return tuple(t_c(arg) for arg in args)
            if type(args) == dict:
                return {k: t_c(arg) for k, arg in args.items()}

        def _wrapper(*args, **kwargs):
            self.comm.send((method.__name__, converter(args), converter(kwargs)), self.remote)
            if delayed:
                return partial(self.comm.recv, source=self.remote)
            return self.comm.recv(source=self.remote)

        return _wrapper


class MpiRPCVecEnv(VecEnv):
    def __init__(self, comm: MPI.Comm, *args, env_type=Env, ep_length=1, **kwargs):
        self.waiting = False
        self.closed = False
        self.env_type = env_type
        self.ep_length = ep_length
        self.worker_size = comm.Get_size() - 1
        self.n_envs = self.worker_size

        self.comm = comm
        self.results = []

        self.pool = joblib.Parallel(n_jobs=self.worker_size)
        self.comm.send(("get_spaces", (), {}), 1)
        observation_space, action_space = self.comm.recv(source=1)

        self.remote_envs = [RPCEnvWrapper(comm, i, *args, **kwargs) for i in range(1, 1 + self.worker_size)]
        self.indices = []
        super().__init__(self.n_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.results = self._await_workers([
            remote.step(actions[i])
            for i, remote in enumerate(self.remote_envs)
        ])
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.waiting = False
        obs, rewards, dones, infos, self.reset_infos = zip(*self.results)
        flat_obs = _flatten_obs(obs, self.observation_space), np.stack(rewards), np.stack(dones), infos
        return flat_obs

    def reset(self) -> VecEnvObs:
        results = self._await_workers([
            remote.reset(self._seeds[env_idx % self.n_envs])
            for env_idx, remote in enumerate(self.remote_envs)
        ])

        obs, self.reset_infos = zip(*results)
        # Seeds are only used once
        self._reset_seeds()
        return _flatten_obs(obs, self.observation_space)

    def buffer_add(self, observations, actions, rewards, episode_starts, values, log_probs) -> None:
        self._await_workers([
            remote.buffer_add(observations[i], actions[i], rewards[i], episode_starts[i], values[i], log_probs[i])
            for i, remote in enumerate(self.remote_envs)
        ])

    def buffer_reset(self) -> None:
        data = self._await_workers([
            remote.buffer_reset()
            for i, remote in enumerate(self.remote_envs)
        ])

    def buffer_compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        """
        Returns the self.returns and self.values of buffer.
        """
        return self._await_workers([
            remote.buffer_compute_returns_and_advantage(last_values[i], dones[i])
            for i, remote in enumerate(self.remote_envs)
        ])

    def buffer_emit_batches(self):
        [
            remote.buffer_emit_batches()
            for i, remote in enumerate(self.remote_envs)
        ]

    def close(self) -> None:
        if self.closed:
            return
        self._await_workers([
            remote.close()
            for remote in self.remote_envs
        ])
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remote_envs]
        outputs = self._await_workers([
            remote.render()
            for remote in self.remote_envs
        ])
        return outputs

    @staticmethod
    def _await_workers(partials):
        return [p() for p in partials]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        return self._await_workers([
            remote.get_attr(attr_name)
            for remote in target_remotes
        ])

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        self._await_workers([
            remote.set_attr(attr_name, value)
            for remote in target_remotes
        ])

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        return self._await_workers([
            remote.env_method(method_name, *method_args, **method_kwargs)
            for remote in target_remotes
        ])

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        return self._await_workers([
            remote.is_wrapped(wrapper_class)
            for remote in target_remotes
        ])

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[RPCEnvWrapper]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remote_envs[i] for i in indices]
