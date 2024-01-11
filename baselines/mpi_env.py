import multiprocessing as mp
import traceback
import warnings
from typing import Any, Callable, List, Optional, Sequence, Type
from typing import Dict

import gymnasium as gym
import numpy as np
from mpi4py import MPI
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from utils import make_env


class StaggeredMPIEnv(VecEnv):
    def __init__(self, comm: MPI.Comm, stagger_count=6):
        self.waiting = False
        self.closed = False
        self.worker_size = comm.Get_size() - 1
        print("Worker size:", self.worker_size)
        self.n_envs = self.worker_size // stagger_count

        self.stagger = -1
        self.stagger_count = stagger_count
        self.comm = comm

        self.comm.send(("get_spaces", None), 1)
        observation_space, action_space = self.comm.recv(source=1)

        self._remotes = list(range(1, comm.Get_size()))
        print(self._remotes)

        super().__init__(self.n_envs, observation_space, action_space)

    @property
    def remotes(self) -> List[int]:
        # The below statement is true by definition
        # assert len(self._remotes) % self.stagger_count == 0
        part = self.worker_size // self.stagger_count
        next_stagger = (self.stagger + 1) % self.stagger_count

        # First time return everything (send all processes initial state)
        if self.stagger == -1:
            return self._remotes

        if next_stagger == 0:
            return self._remotes[self.stagger * part:]
        else:
            return self._remotes[self.stagger * part:next_stagger * part]

    def step_async(self, actions: np.ndarray) -> None:
        remotes = self.remotes
        self.stagger = (self.stagger + 1) % self.stagger_count

        for i, remote in enumerate(remotes):
            self.comm.send(("step", actions[i % len(actions)]), remote)
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [self.comm.recv(source=remote) for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        flat_obs = _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
        return flat_obs

    def reset(self) -> VecEnvObs:
        if self.stagger != -1:
            # Cleanup in-transit messages
            self.step_wait()

        for env_idx, remote in enumerate(self._remotes):
            self.comm.send(("reset", self._seeds[env_idx % self.n_envs]), remote)

        # We receive data from everybody, then drop unnecessary data.
        results = [self.comm.recv(source=remote) for remote in self._remotes][:self.n_envs]
        obs, self.reset_infos = zip(*results)
        # Seeds are only used once
        self._reset_seeds()
        self.stagger = -1
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                self.comm.recv(source=remote)
        for remote in self.remotes:
            self.comm.send(("close", None), remote)
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for remote in self.remotes:
            # gather render return from subprocesses
            self.comm.send(("render", None), remote)
        outputs = [self.comm.recv(source=remote) for remote in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            self.comm.send(("get_attr", attr_name), remote)
        response = [self.comm.recv(source=remote) for remote in target_remotes]
        return response

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            self.comm.send(("set_attr", (attr_name, value)), remote)
        for remote in target_remotes:
            self.comm.recv(source=remote)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            self.comm.send(("env_method", (method_name, method_args, method_kwargs)), remote)
        return [self.comm.recv(source=remote) for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            self.comm.send(("is_wrapped", wrapper_class), remote)
        return [self.comm.recv(source=remote) for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def mpi_worker(comm: MPI.Comm, rank, env_fn_wrapper) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = comm.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                comm.send((observation, reward, done, info, reset_info), 0)
            elif cmd == "reset":
                observation, reset_info = env.reset(seed=data)
                comm.send((observation, reset_info), 0)
            elif cmd == "render":
                comm.send(env.render(), 0)
            elif cmd == "close":
                env.close()
                break
            elif cmd == "get_spaces":
                comm.send((env.observation_space, env.action_space), 0)
            elif cmd == "env_method":
                method = getattr(env, data[0])
                comm.send(method(*data[1], **data[2]), 0)
            elif cmd == "get_attr":
                comm.send(getattr(env, data), 0)
            elif cmd == "set_attr":
                comm.send(setattr(env, data[0], data[1]), 0)  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                comm.send(is_wrapped(env, data), 0)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


def main_mpi(env_conf, seed=0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert (comm.Get_size() > 1)
    try:
        mpi_worker(comm, rank, CloudpickleWrapper(make_env(rank, env_conf, seed=seed)))
    except Exception as e:  # noqa
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main_mpi({})
