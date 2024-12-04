import datetime
import logging
import traceback
import uuid
from os.path import exists
from pathlib import Path

from mpi4py import MPI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from buffer import RemoteMPIRolloutBuffer
from mpi_master_env import MpiRPCVecEnv
from mpi_worker_loop import main_mpi
from utils import make_env


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )


GAE_LAMBDA = 0.95
GAMMA = 0.998


def main():
    setup_logger()
    ep_length = 2048 #* 8

    sess_path = Path(f'/tmp/session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': 100, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': False, 'extra_buttons': False,
        'buffer_size': ep_length,
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank > 0:
        return main_mpi(env_config, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    env = MpiRPCVecEnv(comm, [make_env(0, env_config)], ep_length=ep_length, start_method="fork")
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, save_path=sess_path,
        name_prefix='poke'
    )
    # env_checker.check_env(env)
    learn_steps = 40
    file_name = 'session_995bee40/poke_7667712_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = (comm.Get_size() - 1)
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = (comm.Get_size() - 1)
        model.rollout_buffer.reset()
    else:
        MULTIPLIER = 4
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=128 * MULTIPLIER,
            learning_rate=0.0003 * MULTIPLIER,
            n_epochs=3,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

    model.rollout_buffer = RemoteMPIRolloutBuffer(comm, env)
    model.rollout_buffer.prefetch_factor = 3

    results = env.reset()  # Get how many workers we have.
    print(f"We have {len(results)} workers")
    print(f"Pinging workers for latency info:")
    print(f"0    {[(datetime.datetime.now() - x).total_seconds() for x in env.ping()]}")
    print(f"1    {[(datetime.datetime.now() - x).total_seconds() for x in env.ping()]}")
    print(f"2    {[(datetime.datetime.now() - x).total_seconds() for x in env.ping()]}")
    print(f"3    {[(datetime.datetime.now() - x).total_seconds() for x in env.ping()]}")
    print(f"4    {[(datetime.datetime.now() - x).total_seconds() for x in env.ping()]}")

    model.learn(total_timesteps=ep_length * len(results) * learn_steps, callback=checkpoint_callback)

    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # noqa
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
