import numpy as np
import torch
from mpi4py import MPI
from tqdm import tqdm

SEND_CHUNK_SIZE = 16
N_DATA_SAMPLES_TO_GENERATE = 2 ** 20 // SEND_CHUNK_SIZE  # ~1m


def main_master(comm):
    n_workers = comm.size - 1
    batch_size = 128
    prefetch_factor = 1

    assert N_DATA_SAMPLES_TO_GENERATE % batch_size == 0

    # Prefetch data from workers
    for i in range(0, prefetch_factor * batch_size):
        # Notify all to send message
        current_worker = 1 + (i % n_workers)
        comm.send([0], dest=current_worker)  # Send ack

    bar = tqdm(total=N_DATA_SAMPLES_TO_GENERATE)
    for i in range(0, N_DATA_SAMPLES_TO_GENERATE, batch_size):
        batch = []

        # Request less based on prefetch factor (multiplied by batch count)
        if i < N_DATA_SAMPLES_TO_GENERATE - (batch_size * prefetch_factor):
            # Notify all to send message
            for j in range(0, batch_size):
                # Current worker starts from 1
                current_worker = 1 + ((i + j) % n_workers)
                comm.send([0], dest=current_worker)  # Send ack

        # Receive data from all
        for j in range(0, batch_size):
            # Current worker starts from 1
            current_worker = 1 + ((i + j) % n_workers)
            sample = comm.recv(source=current_worker)  # Recv data

            batch.append(sample)
            bar.update()

        xs = torch.Tensor(np.array(batch))
        # Do something here for fwd backward pass
    # comm.scatter(1, 0)


def main_worker(comm):
    n_workers = (comm.size - 1)

    assert N_DATA_SAMPLES_TO_GENERATE % n_workers == 0
    for i in range(N_DATA_SAMPLES_TO_GENERATE // n_workers):

        ack = comm.recv(source=0)  # Recv ack

        comm.send(np.random.randn(SEND_CHUNK_SIZE, 36, 40), dest=0)  # Send data


def entry():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        main_master(comm)
    else:
        main_worker(comm)


if __name__ == "__main__":
    entry()
