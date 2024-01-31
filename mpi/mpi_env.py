import multiprocessing as mp
import time
import traceback
import types
import warnings
from builtins import function
from typing import Any, Callable, List, Optional, Sequence, Type
from typing import Dict

import gymnasium as gym
import numpy as np
from gym import Env
from mpi4py import MPI
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from red_gym_env import RedGymEnv
from run import MPIRolloutBuffer
from utils import make_env


class RPCEnvWrapperWorker(RedGymEnv):
    def __init__(self):
        super().__init__()
        self.buffer = MPIRolloutBuffer(buffer_size=1024, observation_space=self.observation_space,
                                       action_space=self.action_space)
        self.buffer_add = self.buffer.add
        self.buffer_emit_batches = self.buffer.emit_batches

    def env_method(self, method, *args, **kwargs):
        method = getattr(self, method)
        return method(*args, **kwargs)

    def get_attr(self, prop):
        return getattr(self, prop)

    def set_attr(self, prop, value):
        return setattr(self, prop, value)

    def is_wrapped(self, data):
        return is_wrapped(self, data)


class RPCEnvWrapper(RPCEnvWrapperWorker):
    def __init__(self, comm, remote):  # noqa: we ignore super call because we want a dummy env for RPC
        self.comm = comm
        self.remote = remote

        for method_name in dir(super()):
            prop = super().__getattribute__(method_name)
            if type(prop) == function:
                self.__setattr__(method_name, self._rpc_wrapper(prop))

    def _rpc_wrapper(self, method):
        def _wrapper(*args, **kwargs):
            assert kwargs is None
            self.comm.send((method, args), self.remote)
            return self.comm.recv(source=self.remote)

        return _wrapper


class MpiRPCVecEnv(VecEnv):
    def __init__(self, comm: MPI.Comm, env_type=Env):
        self.waiting = False
        self.closed = False
        self.env_type = env_type
        self.worker_size = comm.Get_size() - 1
        self.n_envs = self.worker_size

        self.stagger = -1
        self.comm = comm
        self.results = []

        self.comm.send(("get_spaces", None), 1)
        observation_space, action_space = self.comm.recv(source=1)

        self.remote_envs = [RPCEnvWrapper(comm=comm, remote=i) for i in range(1, 1 + self.worker_size)]

        super().__init__(self.n_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.results = [remote.step(actions[i % len(actions)]) for i, remote in enumerate(self.remote_envs)]
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.waiting = False
        obs, rewards, dones, infos, self.reset_infos = zip(*self.results)
        flat_obs = _flatten_obs(obs, self.observation_space), np.stack(rewards), np.stack(dones), infos
        return flat_obs

    def reset(self) -> VecEnvObs:
        if self.stagger != -1:
            # Cleanup in-transit messages
            self.step_wait()

        results = [
            remote.reset(self._seeds[env_idx % self.n_envs])
            for env_idx, remote in enumerate(self.remote_envs)
        ]

        obs, self.reset_infos = zip(*results)
        # Seeds are only used once
        self._reset_seeds()
        self.stagger = -1
        return _flatten_obs(obs, self.observation_space)

    def buffer_add(self, observations, actions, rewards, episode_starts, values, log_probs) -> None:
        [
            remote.buffer_add(observations[i], actions[i], rewards[i], episode_starts[i], values[i], log_probs[i])
            for i, remote in enumerate(self.remote_envs)
        ]

    def buffer_emit_batches(self):
        [
            remote.buffer_emit_batches(comm=self.comm)
            for i, remote in enumerate(self.remote_envs)
        ]

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remote_envs:
            remote.close()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remote_envs]
        outputs = [
            remote.render()
            for remote in self.remote_envs
        ]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        return [
            remote.get_attr(attr_name)
            for remote in target_remotes
        ]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.set_attr(attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        return [
            remote.env_method(method_name, *method_args, **method_kwargs)
            for remote in target_remotes
        ]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        return [
            remote.is_wrapped(wrapper_class)
            for remote in target_remotes
        ]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[RPCEnvWrapper]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remote_envs[i] for i in indices]

