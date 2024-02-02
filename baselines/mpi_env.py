import time
import traceback

from mpi4py import MPI
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from mpi_worker_env import RPCEnvWrapperWorker
from utils import make_env


def mpi_worker(comm: MPI.Comm, rank, env_fn_wrapper) -> None:
    env = _patch_env(env_fn_wrapper.var())

    while True:
        while not comm.Iprobe():
            time.sleep(1e-5)
        cmd, data = comm.recv()

        rpc_func = env.__getattribute__(cmd)
        response = rpc_func(data)
        comm.send(response, 0)

        if cmd == "close":
            break


def main_mpi(env_conf, seed=0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert (comm.Get_size() > 1)
    try:
        mpi_worker(comm, rank, CloudpickleWrapper(make_env(rank, env_conf, env_cls=RPCEnvWrapperWorker, seed=seed)))
    except Exception as e:  # noqa
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
