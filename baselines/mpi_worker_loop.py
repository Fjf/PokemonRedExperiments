import os
import time
import traceback

from mpi4py import MPI

from mpi_worker_env import RPCEnvWrapperWorker
from utils import make_env


def mpi_worker(comm: MPI.Comm, env) -> None:
    while True:
        while not comm.Iprobe():
            time.sleep(1e-5)
        cmd, args, kwargs = comm.recv()

        # if comm.rank == 1:
        #     print(f"Executing {cmd}")

        rpc_func = env.__getattribute__(cmd)
        response = rpc_func(*args, **kwargs)
        comm.send(response, 0)

        # if comm.rank == 1:
        #     print("Done")

        if cmd == "close":
            break


def main_mpi(env_conf, seed=0, **kwargs):
    comm = MPI.COMM_WORLD

    # Get core count from SLURM or fall back on max CPUs on machine.
    num_cpu = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
    env = RPCEnvWrapperWorker([make_env(i, env_conf) for i in range(num_cpu)], start_method="fork", **kwargs)
    assert (comm.Get_size() > 1)
    try:
        mpi_worker(comm, env)
    except Exception as e:  # noqa
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
