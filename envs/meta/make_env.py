import gymnasium as gym

from envs.meta.wrappers import VariBadWrapper
from envs.parallel_env_manager import ParallelEnvManager

# In VariBAD, they use on-policy PPO by vectorized env.
# In BOReL, they use off-policy SAC by single env.


def make_env(env_id, episodes_per_task, seed=None, oracle=False, **kwargs):
    """
    kwargs: include n_tasks=num_tasks
    """
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    env = VariBadWrapper(
        env=env,
        episodes_per_task=episodes_per_task,
        oracle=oracle,
    )
    return env


def make_parallel_env(env_id, episodes_per_task, num_workers, seed=None, oracle=False, **kwargs):
    """
    Create a ParallelEnvManager with multiple environment workers.

    Args:
        env_id: Environment ID for gym.make()
        episodes_per_task: Number of episodes per task for VariBadWrapper
        num_workers: Number of parallel environment workers
        seed: Base seed for environments (each worker gets seed + worker_id)
        oracle: Whether to use oracle mode in VariBadWrapper
        **kwargs: Additional arguments for gym.make() (e.g., n_tasks, num_train_tasks, num_eval_tasks)

    Returns:
        ParallelEnvManager instance
    """
    def make_single_env(rank):
        """Factory function to create a single environment for a worker."""
        def _init():
            env_seed = seed + rank if seed is not None else None
            env = make_env(
                env_id=env_id,
                episodes_per_task=episodes_per_task,
                seed=env_seed,
                oracle=oracle,
                **kwargs,
            )
            return env
        return _init

    # Create list of environment factory functions
    env_fns = [make_single_env(i) for i in range(num_workers)]

    # Create ParallelEnvManager with spawn method for better isolation
    parallel_env = ParallelEnvManager(env_fns, start_method='spawn')

    return parallel_env
