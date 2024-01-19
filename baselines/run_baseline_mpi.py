import os
import uuid
from os.path import exists
from pathlib import Path

from mpi4py import MPI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import make_env
from fast_subproc_vec_env import StaggeredSubprocVecEnv
from mpi_env import main_mpi, mpi_worker, StaggeredMPIEnv


def main():
    print(" I am loading")
    stagger_count = 4
    ep_length = 2048 * 4
    max_steps = 2048 * 8

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': max_steps,
        'print_rewards': 100, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank > 0:
        main_mpi(env_config)

    env = StaggeredMPIEnv(comm, stagger_count=stagger_count)
    checkpoint_callback = CheckpointCallback(save_freq=ep_length * stagger_count, save_path=sess_path,
                                             name_prefix='poke')
    # env_checker.check_env(env)
    learn_steps = 1
    file_name = 'session_995bee40/poke_7667712_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length * stagger_count
        model.n_envs = (comm.Get_size() - 1) // stagger_count
        model.rollout_buffer.buffer_size = ep_length * stagger_count
        model.rollout_buffer.n_envs = (comm.Get_size() - 1) // stagger_count
        model.rollout_buffer.reset()
    else:
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            n_steps=ep_length * stagger_count,
            batch_size=512,
            n_epochs=1,
            gamma=0.999
        )

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * comm.Get_size() * 1, callback=checkpoint_callback)

    env.close()


if __name__ == '__main__':
    main()
