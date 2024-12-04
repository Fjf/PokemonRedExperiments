import os
import uuid
from os.path import exists
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import make_env


def main():
    stagger_count = 1
    ep_length = 2048 * 4
    max_steps = 2048 * 8

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': max_steps,
        'print_rewards': 100, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False,
        'buffer_size': ep_length * stagger_count,
    }

    # Get core count from SLURM or fall back on max CPUs on machine.
    num_cpu = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length * stagger_count, save_path=sess_path,
        name_prefix='poke'
    )
    # env_checker.check_env(env)
    learn_steps = 40
    file_name = 'session_995bee40/poke_7667712_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length * stagger_count
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length * stagger_count
        model.rollout_buffer.n_envs = num_cpu
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
        from buffer import MPIRolloutBuffer
        model.rollout_buffer = MPIRolloutBuffer(
            model.rollout_buffer.buffer_size,
            observation_space=model.rollout_buffer.observation_space,
            action_space=model.rollout_buffer.action_space,
            gae_lambda=model.rollout_buffer.gae_lambda,
            gamma=model.rollout_buffer.gamma,
            n_workers=model.rollout_buffer.n_envs
        )

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * stagger_count * num_cpu * 1000, callback=checkpoint_callback)


if __name__ == '__main__':
    main()
