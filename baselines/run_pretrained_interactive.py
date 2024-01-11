import os
import uuid
from pathlib import Path

from stable_baselines3 import PPO

from utils import make_env

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2 ** 23

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config, seed=int(os.environ.get("SEED", 0)))()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # env_checker.check_env(env)
    file_name = 'session_4da05e87_main_good/poke_439746560_steps'

    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()
