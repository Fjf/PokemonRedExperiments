import os
from os.path import exists
import uuid
from os.path import exists
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from fast_subproc_vec_env import StaggeredSubprocVecEnv
from red_gym_env import RedGymEnv


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def main():
    stagger_count = 4
    ep_length = 2048 * 8 * stagger_count

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': 100, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False
    }

    # Get core count from SLURM or fall back on max CPUs on machine.
    num_cpu = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))

    env = StaggeredSubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)], stagger_count=stagger_count)

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                             name_prefix='poke')
    # env_checker.check_env(env)
    learn_steps = 40
    file_name = 'wakuwaku'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            'CnnPolicy',
            env,
            verbose=0,
            n_steps=ep_length,
            batch_size=512,
            n_epochs=1,
            gamma=0.999
        )

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=checkpoint_callback)


if __name__ == '__main__':
    main()
