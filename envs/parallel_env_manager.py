"""
Parallel Environment Manager for Multi-Process Rollout Collection

This module provides a multiprocessing-based environment manager that spawns
multiple worker processes, each running its own training environment. It allows
for parallel rollout collection compatible with the learner.py training pipeline.
"""

import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Callable, Any, Dict, List, Tuple
import traceback


class EnvWorker(Process):
    """
    Worker process that manages a single environment instance.
    Communicates with the main process via pipes.
    """

    def __init__(
        self,
        remote: Connection,
        env_fn: Callable,
        worker_id: int,
    ):
        """
        Args:
            remote: Pipe connection to communicate with main process
            env_fn: Callable that creates and returns an environment instance
            worker_id: Unique identifier for this worker
        """
        super().__init__(daemon=True)
        self.remote = remote
        self.env_fn = env_fn
        self.worker_id = worker_id
        self.env = None

    def run(self):
        """Main loop that listens for commands from the main process."""
        try:
            # Initialize environment in worker process
            self.env = self.env_fn()

            while True:
                try:
                    cmd, data = self.remote.recv()
                except EOFError:
                    break

                if cmd == 'reset':
                    # data contains task information (can be None)
                    obs = self.env.reset(**data)
                    self.remote.send(('obs', obs))

                elif cmd == 'step':
                    # data contains action
                    obs, reward, done, info = self.env.step(**data)

                    # Check if environment has is_goal_state method (for meta-learning envs)
                    if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'is_goal_state'):
                        info['is_goal_state'] = self.env.unwrapped.is_goal_state()

                    self.remote.send(('transition', (obs, reward, done, info)))

                elif cmd == 'get_attr':
                    # Get environment attribute
                    attr_name = data['attr']
                    if attr_name in dir(self.env):
                        attr = getattr(self.env, attr_name)
                    else:
                        assert attr_name in dir(self.env.unwrapped)
                        attr = getattr(self.env.unwrapped, attr_name)
                    # Handle nested attributes (e.g., 'unwrapped.tasks')
                    for subattr in data.get('subattrs', []):
                        attr = getattr(attr, subattr)
                    self.remote.send(('attr', attr))

                elif cmd == 'close':
                    self.env.close()
                    self.remote.close()
                    break

                elif cmd == 'seed':
                    seed_value = data['seed']
                    self.env.seed(seed_value)
                    if hasattr(self.env, 'action_space'):
                        self.env.action_space.np_random.seed(seed_value)
                    self.remote.send(('done', None))

                else:
                    raise NotImplementedError(f"Command {cmd} not implemented")

        except Exception as e:
            print(f"Worker {self.worker_id} encountered error: {e}")
            traceback.print_exc()
        finally:
            if self.env is not None:
                self.env.close()


class ParallelEnvManager:
    """
    Manages multiple environment workers running in parallel processes.

    This class provides an interface similar to a single environment but
    runs multiple environments in parallel, allowing for efficient data collection.
    """

    def __init__(
        self,
        env_fns: List[Callable],
        start_method: Optional[str] = None,
    ):
        """
        Args:
            env_fns: List of callables, each returning an environment instance
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
                         If None, uses the default for the platform
        """
        self.num_envs = len(env_fns)
        self.env_fns = env_fns
        self.waiting = False
        self.closed = False

        # Set multiprocessing start method if specified
        if start_method is not None:
            ctx = mp.get_context(start_method)
        else:
            ctx = mp.get_context()

        # Create pipes for communication with workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])

        # Start worker processes
        self.workers = [
            EnvWorker(work_remote, env_fn, idx)
            for idx, (work_remote, env_fn) in enumerate(zip(self.work_remotes, env_fns))
        ]

        for worker in self.workers:
            worker.start()

        # Close work remotes in main process (only workers should use them)
        for work_remote in self.work_remotes:
            work_remote.close()

        # Get environment properties from first worker
        # This assumes all environments have the same properties
        self._fetch_env_properties()

    def _fetch_env_properties(self):
        """Fetch common environment properties from the first worker."""
        # Get observation space
        self.remotes[0].send(('get_attr', {'attr': 'observation_space'}))
        _, self.observation_space = self.remotes[0].recv()

        # Get action space
        self.remotes[0].send(('get_attr', {'attr': 'action_space'}))
        _, self.action_space = self.remotes[0].recv()

        # Try to get n_tasks (for meta environments)
        self.remotes[0].send(('get_attr', {'attr': 'n_tasks'}))
        _, self.n_tasks = self.remotes[0].recv()

        # Try to get horizon_bamdp (for VariBadWrapper)
        try:
            self.remotes[0].send(('get_attr', {'attr': 'horizon_bamdp'}))
            _, self.horizon_bamdp = self.remotes[0].recv()
        except:
            self.horizon_bamdp = None

        # Try to get unwrapped for accessing underlying environment
        try:
            self.remotes[0].send(('get_attr', {'attr': 'unwrapped'}))
            _, unwrapped = self.remotes[0].recv()
            # Store unwrapped properties we might need
            if hasattr(unwrapped, '_max_episode_steps'):
                self._max_episode_steps = unwrapped._max_episode_steps
        except:
            pass

    def reset(self, env_idx: int = 0, **kwargs) -> np.ndarray:
        """
        Reset a specific environment.

        Args:
            env_idx: Index of the environment to reset (0 to num_envs-1)
            task: Optional task identifier to reset the environment with

        Returns:
            Initial observation from the environment
        """
        if env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")

        assert isinstance(kwargs, dict)
        self.remotes[env_idx].send(('reset', kwargs))
        msg_type, obs = self.remotes[env_idx].recv()
        assert msg_type == 'obs'
        return obs

    def step(self, env_idx: int, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in a specific environment.

        Args:
            env_idx: Index of the environment to step
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")

        self.remotes[env_idx].send(('step', {'action': action}))
        msg_type, transition = self.remotes[env_idx].recv()
        assert msg_type == 'transition'
        return transition

    def reset_all(self, tasks: Optional[List[Optional[int]]] = None) -> List[np.ndarray]:
        """
        Reset all environments in parallel.

        Args:
            tasks: Optional list of task identifiers, one per environment
                  If None, resets all environments without task specification

        Returns:
            List of initial observations, one per environment
        """
        if tasks is None:
            tasks = [None] * self.num_envs
        elif len(tasks) != self.num_envs:
            raise ValueError(f"tasks length {len(tasks)} must match num_envs {self.num_envs}")

        # Send reset commands to all workers
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset', {'task': task}))

        # Collect observations from all workers
        observations = []
        for remote in self.remotes:
            msg_type, obs = remote.recv()
            assert msg_type == 'obs'
            observations.append(obs)

        return observations

    def step_all(self, actions: List[np.ndarray]) -> Tuple[List, List, List, List]:
        """
        Take steps in all environments in parallel.

        Args:
            actions: List of actions, one per environment

        Returns:
            Tuple of (observations, rewards, dones, infos), each as a list
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"actions length {len(actions)} must match num_envs {self.num_envs}")

        # Send step commands to all workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', {'action': action}))

        # Collect transitions from all workers
        observations, rewards, dones, infos = [], [], [], []
        for remote in self.remotes:
            msg_type, (obs, reward, done, info) = remote.recv()
            assert msg_type == 'transition'
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return observations, rewards, dones, infos

    def get_attr(self, attr_name: str, env_idx: int = 0, subattrs: Optional[List[str]] = None):
        """
        Get an attribute from a specific environment.

        Args:
            attr_name: Name of the attribute to get
            env_idx: Index of the environment to query
            subattrs: Optional list of sub-attributes to access (e.g., ['unwrapped', 'tasks'])

        Returns:
            The requested attribute value
        """
        if subattrs is None:
            subattrs = []
        self.remotes[env_idx].send(('get_attr', {'attr': attr_name, 'subattrs': subattrs}))
        msg_type, attr = self.remotes[env_idx].recv()
        assert msg_type == 'attr'
        return attr

    def seed(self, seeds: List[int]):
        """
        Seed all environments.

        Args:
            seeds: List of seed values, one per environment
        """
        if len(seeds) != self.num_envs:
            raise ValueError(f"seeds length {len(seeds)} must match num_envs {self.num_envs}")

        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', {'seed': seed}))

        # Wait for all seeds to be set
        for remote in self.remotes:
            msg_type, _ = remote.recv()
            assert msg_type == 'done'

    def close(self):
        """Close all worker processes."""
        if self.closed:
            return

        # Send close command to all workers
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except:
                pass

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        # Close all remotes
        for remote in self.remotes:
            remote.close()

        self.closed = True

    def __del__(self):
        """Cleanup when object is deleted."""
        if not self.closed:
            self.close()

    @property
    def unwrapped(self):
        """
        Proxy property to access unwrapped environment.
        Note: This returns a proxy object that forwards attribute access to the first worker.
        """
        return UnwrappedProxy(self)


class UnwrappedProxy:
    """
    Proxy object for accessing unwrapped environment attributes.
    This forwards attribute requests to the first worker process.
    """

    def __init__(self, parallel_env: ParallelEnvManager):
        self._parallel_env = parallel_env

    def __getattr__(self, name):
        return self._parallel_env.get_attr('unwrapped', env_idx=0, subattrs=[name])

    def is_goal_state(self):
        """Special handling for is_goal_state method (if environment has it)."""
        # This would need to be called on a specific environment during rollout
        # For now, raise an error as this should be handled differently
        raise NotImplementedError(
            "is_goal_state() should be called on a specific environment during rollout, "
            "not on the parallel manager. Use get_attr('unwrapped', subattrs=['is_goal_state']) "
            "if you need to check this."
        )
