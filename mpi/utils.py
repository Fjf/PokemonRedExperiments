from stable_baselines3.common.utils import set_random_seed

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
        env.reset(seed=(seed + rank), rank=rank)
        return env

    set_random_seed(seed)
    return _init
