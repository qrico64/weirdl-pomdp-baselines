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
                    try:
                        obs = self.env.reset(**data)
                    except Exception:
                        traceback.print_exc()
                    self.remote.send(('obs', obs))

                elif cmd == 'step':
                    # data contains action
                    try:
                        obs, reward, done, info = self.env.step(**data)

                        # Check if environment has is_goal_state method (for meta-learning envs)
                        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'is_goal_state'):
                            info['is_goal_state'] = self.env.unwrapped.is_goal_state()
                    except Exception:
                        traceback.print_exc()

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
                    try:
                        seed_value = data['seed']
                        self.env.seed(seed_value)
                        if hasattr(self.env, 'action_space'):
                            self.env.action_space.np_random.seed(seed_value)
                    except Exception:
                        traceback.print_exc()

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
        self.close()

