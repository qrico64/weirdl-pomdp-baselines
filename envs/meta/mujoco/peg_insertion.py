import numpy as np
from typing import Literal
from utils import logger, helpers
import metaworld
import gymnasium as gym
import imageio
import os
import cv2
from torchkit import pytorch_utils as ptu

from .core.serializable import Serializable

from metaworld.policies import SawyerPegInsertionSideV3Policy


def init_tasks_deterministic_random(n_tasks, low, high):
    xl = int(np.ceil(np.sqrt(n_tasks)))
    yl = int(np.ceil(n_tasks / xl))
    xopts = np.linspace(low[0], high[0], num=xl, endpoint=True)
    yopts = np.linspace(low[1], high[1], num=yl, endpoint=True)

    goals = np.zeros((n_tasks, 6), dtype=np.float32)
    goals[:,0] = xopts[np.arange(n_tasks) // yl]
    goals[:,1] = yopts[np.arange(n_tasks) % yl]
    goals[:,2] = low[2]
    return goals

class PegInsertionEnv(gym.Env, Serializable):
    def __init__(
        self,
        task={},
        num_train_tasks:int=3,
        num_eval_tasks:int=20,
        max_episode_steps=150,
        task_mode: Literal["fixed", "random_peg", "random_target", "random_both"] = "fixed",
        goal_conditioning: str = "no",
        goal_noise_magnitude: float = 0,
        goal_noise_type: Literal["normal", "uniform", "constrained_normal"] = "normal",
        infinite_tasks: Literal["no", "yes"] = "no",
        normalize_kwarg: bool = False,
        seed: int = None,
        **kwargs
    ):
        # Initialize Serializable to support pickling for multiprocessing
        Serializable.quick_init(self, locals())

        assert seed is not None, f"{seed}"
        assert len(kwargs) == 0
        self.seed = seed
        self._max_episode_steps = max_episode_steps
        self.task_mode = task_mode
        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.n_tasks = self.num_train_tasks + self.num_eval_tasks
        self.goal_conditioning = goal_conditioning
        self.goal_conditioning_view = ["no", "yes_target", "yes_peg", "yes_both"]
        self.goal_conditioning_view += [cond + "/reward" for cond in self.goal_conditioning_view]
        self.goal_noise_magnitude = goal_noise_magnitude
        self.goal_noise_type = goal_noise_type
        self.infinite_tasks = infinite_tasks
        self._goal_noise = 0.0
        self.normalize_kwarg = normalize_kwarg
        self.env = gym.make('Meta-World/MT1', env_name='peg-insert-side-v3', render_mode='rgb_array', seed=seed)
        self.env.unwrapped.seed(seed)

        self.action_space = self.env.action_space
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        L = {
            'no': 0,
            'yes_peg': 3,
            'yes_target': 3,
            'yes_both': 6,
        }[self.goal_conditioning]
        low = np.full((self.env.observation_space.shape[0] + L,), -np.inf, dtype=self.env.observation_space.dtype)
        high = np.full((self.env.observation_space.shape[0] + L,), np.inf, dtype=self.env.observation_space.dtype)
        self.observation_space = gym.spaces.Box(low, high, dtype=self.env.observation_space.dtype)

        logger.log("\n****** Creating PegInsertionEnv Environment ******")
        logger.log(f"n_tasks: {self.n_tasks}")
        logger.log(f"num_train_tasks: {self.num_train_tasks}")
        logger.log(f"num_eval_tasks: {self.num_eval_tasks}")
        logger.log(f"task_mode: {self.task_mode}")
        logger.log(f"goal_conditioning: {self.goal_conditioning}")
        logger.log(f"goal_conditioning_view: {self.goal_conditioning_view}")
        logger.log(f"goal_noise_magnitude: {self.goal_noise_magnitude}")
        logger.log(f"goal_noise_type: {self.goal_noise_type}")
        logger.log(f"infinite_tasks: {self.infinite_tasks}")
        logger.log(f"normalize_kwarg: {self.normalize_kwarg}")
        logger.log(f"observation_space: {self.observation_space.shape}")
        logger.log(f"action_space: {self.action_space.shape}")
        logger.log("****** Created PegInsertionEnv Environment ******\n")

        self._last_obs = None
        self._last_success = False
        self._last_reward = 0.0

        super(PegInsertionEnv, self).__init__()
        self.init_consts()
        self.init_tasks()
        assert os.environ['MUJOCO_GL'] == 'egl'
    
    def init_consts(self):
        self.bounds = {}
        self.bounds['peg_init_pos'] = self.env.unwrapped.obj_init_pos
        self.bounds['peg_bounds'] = {
            'low': self.env.unwrapped._random_reset_space.low[:3],
            'high': self.env.unwrapped._random_reset_space.high[:3]
        }
        self.bounds['target_bounds'] = {
            'low': self.env.unwrapped.goal_space.low,
            'high': self.env.unwrapped.goal_space.high
        }
        self.bounds['default_peg_reset_pos'] = np.array([0.18491799, 0.66787545, 0.02], dtype=np.float32)
        self.bounds['default_target_reset_pos'] = np.array([-0.28711311, 0.4484228, 0.12945647], dtype=np.float32)
    
    def init_tasks(self):
        self._task = None
        self._goal = None
        if self.infinite_tasks == "yes":
            self.train_goals = np.stack([self.train_task_distribution() for _ in range(self.num_train_tasks)], axis=0)
            self.eval_goals = np.stack([self.train_task_distribution() for _ in range(self.num_eval_tasks)], axis=0)
        elif self.task_mode == "fixed":
            self.train_goals = np.tile(np.concatenate([self.bounds['default_peg_reset_pos'], self.bounds['default_target_reset_pos']]), (self.num_train_tasks, 1))
            self.eval_goals = np.tile(np.concatenate([self.bounds['default_peg_reset_pos'], self.bounds['default_target_reset_pos']]), (self.num_eval_tasks, 1))
        elif self.task_mode == "random_peg":
            goals_peg = init_tasks_deterministic_random(self.n_tasks, self.bounds['peg_bounds']['low'], self.bounds['peg_bounds']['high'])
            goals = np.concatenate([goals_peg, np.tile(self.bounds['default_target_reset_pos'][None,:], (self.n_tasks, 1))], axis=1)

            assignment = np.zeros((self.n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, self.n_tasks, self.num_eval_tasks, endpoint=False)).astype(np.int32)] = True

            self.train_goals = goals[np.logical_not(assignment)]
            self.eval_goals = goals[assignment]
        elif self.task_mode == "random_target":
            goals_target = init_tasks_deterministic_random(self.n_tasks, self.bounds['target_bounds']['low'], self.bounds['target_bounds']['high'])
            goals = np.concatenate([np.tile(self.bounds['default_peg_reset_pos'][None,:], (self.n_tasks, 1)), goals_target], axis=1)

            assignment = np.zeros((self.n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, self.n_tasks, self.num_eval_tasks, endpoint=False)).astype(np.int32)] = True

            self.train_goals = goals[np.logical_not(assignment)]
            self.eval_goals = goals[assignment]
        elif self.task_mode == "random_both":
            goals_peg = init_tasks_deterministic_random(self.n_tasks, self.bounds['peg_bounds']['low'], self.bounds['peg_bounds']['high'])
            goals_target = init_tasks_deterministic_random(self.n_tasks, self.bounds['target_bounds']['low'], self.bounds['target_bounds']['high'])
            goals = np.concatenate([goals_peg, goals_target], axis=1)

            assignment = np.zeros((self.n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, self.n_tasks, self.num_eval_tasks, endpoint=False)).astype(np.int32)] = True

            self.train_goals = goals[np.logical_not(assignment)]
            self.eval_goals = goals[assignment]
        else:
            raise NotImplementedError()
        self.goals = np.concatenate([self.train_goals, self.eval_goals], axis=0)
        self.tasks = [{'peg_pos': self.goals[i][:3], 'target_pos': self.goals[i][3:]} for i in range(self.n_tasks)]

    def _set_peg_pos(self, peg_pos):
        assert isinstance(peg_pos, np.ndarray) and peg_pos.shape == (3,)
        self.env.unwrapped._set_obj_xyz(peg_pos)

    def _set_target_pos(self, target_pos):
        assert isinstance(target_pos, np.ndarray) and target_pos.shape == (3,)
        self.env.unwrapped._target_pos = target_pos
        self.env.unwrapped.goal = target_pos
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(seed=self.seed)
        self._last_success = False
        self._last_reward = 0.0
        self._set_peg_pos(self._task['peg_pos'])
        self._set_target_pos(self._task['target_pos'])
        obs, info["obs"] = self._get_obs2(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_success = info.get('success', False)
        self._last_reward = reward
        self._last_obs = obs
        obs, info["obs"] = self._get_obs2(obs)
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _append_obs_raw(self, obs, return_obs_type):
        if '/' in return_obs_type:
            cur_goal_condition, cur_reward_condition = return_obs_type.split('/')
        else:
            cur_goal_condition, cur_reward_condition = return_obs_type, None
        
        target_goal = self._goal + self._goal_noise
        if cur_goal_condition == "yes_peg":
            obs = np.concatenate([obs, target_goal[:3]], axis=0)
        elif cur_goal_condition == "yes_target":
            obs = np.concatenate([obs, target_goal[3:]], axis=0)
        elif cur_goal_condition == "yes_both":
            obs = np.concatenate([obs, target_goal], axis=0)
        elif cur_goal_condition == "no":
            obs = np.copy(obs)
        else:
            raise NotImplementedError(f"Unidentified goal conditioning: {cur_goal_condition}")
        
        if cur_reward_condition is None:
            pass
        elif cur_reward_condition == "reward":
            obs = np.concatenate([obs, [self._last_reward]], axis=0)
        else:
            raise NotImplementedError(f"Unidentified reward conditioning: {cur_reward_condition}")
        
        return obs
    
    def _get_obs2(self, obs):
        info = {k: self._append_obs_raw(obs, return_obs_type=k) for k in self.goal_conditioning_view}
        return self._append_obs_raw(obs, return_obs_type=self.goal_conditioning), info

    def _get_obs(self):
        obs = self.env.unwrapped._get_obs()
        return self._append_obs_raw(obs, return_obs_type=self.goal_conditioning)

    def train_task_distribution(self):
        if self.task_mode == "fixed":
            peg = self.bounds['default_peg_reset_pos']
            target = self.bounds['default_target_reset_pos']
        elif self.task_mode == "random_peg":
            peg = np.random.uniform(self.bounds['peg_bounds']['low'], self.bounds['peg_bounds']['high'])
            target = self.bounds['default_target_reset_pos']
        elif self.task_mode == "random_target":
            peg = self.bounds['default_peg_reset_pos']
            target = np.random.uniform(self.bounds['target_bounds']['low'], self.bounds['target_bounds']['high'])
        elif self.task_mode == "random_both":
            peg = np.random.uniform(self.bounds['peg_bounds']['low'], self.bounds['peg_bounds']['high'])
            target = np.random.uniform(self.bounds['target_bounds']['low'], self.bounds['target_bounds']['high'])
        else:
            raise NotImplementedError()

        return np.concatenate([peg, target], axis=0)
    
    def reset_task(self, goal, override_task=None):
        if goal is not None:
            self._goal = goal
            self._task = {'peg_pos': self._goal[:3], 'target_pos': self._goal[3:]}
        else:
            self._goal = self.train_task_distribution()
            self._task = {'peg_pos': self._goal[:3], 'target_pos': self._goal[3:]}
        
        if self.goal_noise_type == "normal":
            self._goal_noise = np.random.randn(6) * self.goal_noise_magnitude
        elif self.goal_noise_type == "uniform":
            self._goal_noise = np.random.uniform(-1, 1, (6,)) * self.goal_noise_magnitude
        elif self.goal_noise_type == "constrained_normal":
            self._goal_noise = np.random.randn(6) * self.goal_noise_magnitude
            self._goal_noise = np.clip(self._goal_noise, -self.goal_noise_magnitude, self.goal_noise_magnitude)
        else:
            self._goal_noise = 0.0
        
        if override_task is not None:
            assert isinstance(override_task, np.ndarray)
            self._goal = override_task
            self._task = {'peg_pos': self._goal[:3], 'target_pos': self._goal[3:]}

        self.reset()

    def render(self):
        try:
            frame = self.env.render()
            frame = cv2.resize(frame, (400, 400, 3))
            return frame
        except (AttributeError, ImportError, RuntimeError) as e:
            # Fallback to a blank frame if rendering fails (common in headless environments)
            logger.log(f"Warning: Rendering failed with error: {e}. Returning blank frame.")
            return np.zeros((400, 400, 3), dtype=np.uint8)

    # def render_pos(self) -> np.ndarray:
    #     return self._last_obs[:3] # TODO!!

    def annotation(self) -> str:
        info = {
            '_goal': ptu.format_array_3dec(self._goal),
            '_goal_noise': ptu.format_array_3dec(self._goal_noise),
        }
        return str(info)

    def is_goal_state(self):
        """
        Check if the peg has successfully reached the target position.
        Uses the success value from the last step's info dict.
        """
        return self._last_success

    def __getstate__(self):
        """Save only initialization parameters for serialization."""
        # Return initialization parameters that can recreate the environment
        return {
            'num_train_tasks': self.num_train_tasks,
            'num_eval_tasks': self.num_eval_tasks,
            'max_episode_steps': self._max_episode_steps,
            'task_mode': self.task_mode,
            'goal_conditioning': self.goal_conditioning,
            'goal_noise_magnitude': self.goal_noise_magnitude,
            'goal_noise_type': self.goal_noise_type,
            'infinite_tasks': self.infinite_tasks,
            'normalize_kwarg': self.normalize_kwarg,
            'seed': self.seed,
        }

    def __setstate__(self, state):
        """Recreate the environment from saved parameters."""
        # Reinitialize the environment with saved parameters
        self.__init__(**state)


class MetaWorldExpertPolicy:
    """
    Wrapper around MetaWorld's deterministic expert policy to match the API
    expected by learner.py for base models.
    """
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        assert self.obs_dim == 39
        self.action_dim = action_dim
        self._expert = SawyerPegInsertionSideV3Policy()
        self._expert_env = None

    def to(self, device):
        """No-op for compatibility with PyTorch models."""
        return self

    def eval(self):
        """No-op for compatibility with PyTorch models."""
        return self

    def parameters(self):
        """Return empty list for compatibility with freezing logic."""
        return []

    def load_state_dict(self, state_dict):
        """No-op for compatibility with model syncing."""
        pass

    def state_dict(self):
        """Return empty dict for compatibility with model syncing."""
        return {}

    def get_initial_info(self):
        """Return initial action and reward for Memory agents."""
        import torch
        from torchkit import pytorch_utils as ptu
        action = ptu.zeros((1, self.action_dim))
        reward = ptu.zeros((1, 1))
        return action, reward

    def act(self, prev_actions=None, obs=None, rewards=None, lengths=None,
            deterministic=False, nominals=None, base_actions=None, **kwargs):
        """
        Produce actions using the MetaWorld expert policy.

        For Transformer architecture:
        - obs: (L, batch_size, obs_dim) - sequence of observations
        - Returns: (batch_size, action_dim) - actions for the last timestep

        For Memory/Markov architecture:
        - obs: (batch_size, obs_dim) - current observation
        - Returns: (batch_size, action_dim) - actions
        """
        import torch
        from torchkit import pytorch_utils as ptu

        # Handle both Transformer (sequence) and Markov (single obs) inputs
        if obs.dim() == 3:  # Transformer: (L, batch_size, obs_dim)
            # Get the last observation for each sequence in the batch
            # lengths tells us the actual length of each sequence
            obs = obs[:,:,:self.obs_dim]
            batch_size = obs.shape[1]
            actions = []
            for i in range(batch_size):
                if lengths is not None:
                    # Get observation at position lengths[i]-1 (last valid timestep)
                    last_obs = obs[lengths[i]-1, i, :].cpu().numpy()
                else:
                    # Get last observation
                    last_obs = obs[-1, i, :].cpu().numpy()
                action = self._expert.get_action(last_obs)
                actions.append(action)
            actions = ptu.from_numpy(np.array(actions, dtype=np.float32))
        elif obs.dim() == 2:  # Markov/Memory: (batch_size, obs_dim)
            obs = obs[:,:self.obs_dim]
            batch_size = obs.shape[0]
            actions = []
            for i in range(batch_size):
                obs_np = obs[i, :].cpu().numpy()
                action = self._expert.get_action(obs_np)
                actions.append(action)
            actions = ptu.from_numpy(np.array(actions, dtype=np.float32))
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        # Return format matches policy models: (actions, values, action_log_probs, dist_entropy)
        # We only need actions for the base model, so return None for the rest
        return actions, None, None, None

