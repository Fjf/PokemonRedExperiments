import time

from mpi4py import MPI
from stable_baselines3 import PPO

from buffer import RemoteMPIRolloutBuffer, N_DATA_SAMPLES_TO_GENERATE
from mpi_env import MpiRPCVecEnv, RPCEnvWrapperWorker
from red_gym_env import RedGymEnv



def main_master(comm):
    n_workers = comm.size - 1
    batch_size = 128
    prefetch_factor = 8
    steps = 2048
    n_epochs = 40

    assert N_DATA_SAMPLES_TO_GENERATE % batch_size == 0
    env = MpiRPCVecEnv(comm=comm, env_type=RedGymEnv)
    buffer = RemoteMPIRolloutBuffer(comm=comm, prefetch_factor=prefetch_factor)

    ppo = PPO()
    ppo.rollout_buffer = buffer

    for epoch in range(n_epochs):

        # observation = env.reset()
        # for i in range(env.max_steps):
        #     # TODO: Initialize model
        #     actions, values, log_probs = ppo.policy.forward(torch.from_numpy(observation))
        #     ppo.collect_rollouts()
        #
        #     observation, reward, done, information = env.step(actions=actions.numpy())
        #
        #     # TODO: Get these values from PPO
        #     env.buffer_add(observation, actions, reward, episode_starts, values, log_probs)

        # Inform all workers to emit all batches they have
        env.buffer_emit_batches()

        # This will update our model?
        ppo.train()

        # comm.scatter(1, 0)


def main_worker(comm):
    env = RPCEnvWrapperWorker(config={})

    while True:
        while not comm.Iprobe():
            time.sleep(1e-5)
        cmd, data = comm.recv()

        rpc_func = env.__getattribute__(cmd)
        response = rpc_func(data)
        comm.send(response, 0)

        if cmd == "close":
            break


def entry():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        main_master(comm)
    else:
        main_worker(comm)


if __name__ == "__main__":
    entry()
