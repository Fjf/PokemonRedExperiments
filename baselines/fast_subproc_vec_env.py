import multiprocessing as mp
import warnings
from typing import Any, Callable, List, Optional, Sequence, Type, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _worker(
        remote: mp.connection.Connection,
        parent_remote: mp.connection.Connection,
        env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                observation, reset_info = env.reset()
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class StaggeredSubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, stagger_count=1):
        self.waiting = False
        self.closed = False

        self.n_envs = len(env_fns) // stagger_count

        print("Worker size", self.n_envs)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        print("Startmethod", start_method)
        ctx = mp.get_context(start_method)


        self.stagger = -1
        self.stagger_count = stagger_count

        self._remotes, self._work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs * stagger_count)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self._work_remotes, self._remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            # pytype: disable=attribute-error
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            # pytype: enable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(self.n_envs, observation_space, action_space)

    @property
    def remotes(self):
        # The below statement is true by definition
        # assert len(self._remotes) % self.stagger_count == 0
        part = len(self._remotes) // self.stagger_count
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

        assert len(actions) == len(remotes), "Action count does not equal simulation count."
        for i, remote in enumerate(remotes):
            remote.send(("step", actions[i]))
        self.waiting = True

    def step_wait(self) -> List:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def reset(self) -> List:
        # if self.stagger != -1:
        #     # Cleanup in-transit messages
        #     self.step_wait()

        for env_idx, remote in enumerate(self._remotes):
            remote.send(("reset", None))

        # We receive data from everybody, then drop unnecessary data.
        results = [remote.recv() for remote in self._remotes][:self.n_envs]
        return results

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
