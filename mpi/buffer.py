from typing import Generator

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from tqdm import tqdm

SEND_CHUNK_SIZE = 16
N_DATA_SAMPLES_TO_GENERATE = 2 ** 20 // SEND_CHUNK_SIZE  # ~1m


class MPIRolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            gae_lambda: float = 1,
            gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.pos = 0

        self.action_space = action_space
        self.observation_space = observation_space
        self.action_dim = get_action_dim(action_space)
        self.obs_shape = get_obs_shape(observation_space)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.episode_starts = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.generator_ready = False

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape(self.obs_shape)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape(self.action_dim)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = BaseBuffer.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        for i in range(self.buffer_size):
            yield self._get_samples(indices[i])

    def _get_samples(
            self,
            batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(np.array, data)))

    def emit_batches(self, comm):
        for sample in self.get():
            ack = comm.recv(source=0)  # Recv request to send data
            comm.send(sample, dest=0)  # Send data


class RemoteMPIRolloutBuffer:
    def __init__(self, comm, prefetch_factor=1):
        self.prefetch_factor = prefetch_factor
        self.n_workers = comm.size - 1
        self.comm = comm

    def reset(self):
        for i in range(self.n_workers):
            self.comm.send([0], dest=i)

    def get(self, batch_size=32):
        # Prefetch data from workers
        for i in range(0, self.prefetch_factor * batch_size):
            # Notify all to send message
            current_worker = 1 + (i % self.n_workers)
            self.comm.send([0], dest=current_worker)  # Send ack

        bar = tqdm(total=N_DATA_SAMPLES_TO_GENERATE)
        for i in range(0, N_DATA_SAMPLES_TO_GENERATE, batch_size):
            batch = []

            # Request less based on prefetch factor (multiplied by batch count)
            if i < N_DATA_SAMPLES_TO_GENERATE - (batch_size * self.prefetch_factor):
                # Notify all to send message
                for j in range(0, batch_size):
                    # Current worker starts from 1
                    current_worker = 1 + (((i + self.prefetch_factor * batch_size) + j) % self.n_workers)
                    self.comm.send([0], dest=current_worker)  # Send ack

            # Receive data from all
            for j in range(0, batch_size):
                # Current worker starts from 1
                current_worker = 1 + ((i + j) % self.n_workers)
                sample = self.comm.recv(source=current_worker)  # Recv data

                batch.append(sample)
                bar.update()

            yield np.array(batch)
