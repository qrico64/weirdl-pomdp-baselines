import numpy as np
from typing import Literal
from utils import logger
import gymnasium as gym

from .ant_multitask_base import MultitaskAntEnv
from utils import helpers


class AntDirEnv(MultitaskAntEnv):
    """
    AntDir: forward_backward=True (unlimited tasks) from on-policy varibad code
    AntDir2D: forward_backward=False (limited tasks) from off-policy varibad code
    """

    def __init__(
        self,
        task={},
        num_train_tasks:int=3,
        num_eval_tasks:int=20,
        max_episode_steps=200,
        task_mode: Literal["circle", "circle_down_up", "circle_left_right_up_down", "circle_down_quarter"] = "circle",
        reward_conditioning: Literal["no", "yes"] = "no",
        goal_conditioning: Literal["no", "yes", "fixed_noise", "yes_relative", "yes_relative_noise"] = "no",
        goal_noise_magnitude: float = 0,
        goal_noise_type: Literal["normal", "uniform", "constrained_normal"] = "normal",
        infinite_tasks: Literal["no", "yes"] = "no",
        normalize_kwarg: bool = False,
        forward_backward=True,
        **kwargs
    ):
        self.forward_backward = forward_backward
        self._max_episode_steps = max_episode_steps
        self.task_mode = task_mode
        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.reward_conditioning = reward_conditioning
        self.goal_conditioning = goal_conditioning
        self.goal_noise_magnitude = goal_noise_magnitude
        self.goal_noise_type = goal_noise_type
        self.infinite_tasks = infinite_tasks
        self._goal_noise = 0.0
        self.normalize_kwarg = normalize_kwarg

        logger.log("\n****** Creating AntDir Environment ******")
        logger.log(f"num_train_tasks: {self.num_train_tasks}")
        logger.log(f"num_eval_tasks: {self.num_eval_tasks}")
        logger.log(f"task_mode: {self.task_mode}")
        logger.log(f"goal_conditioning: {self.goal_conditioning}")
        logger.log(f"goal_noise_magnitude: {self.goal_noise_magnitude}")
        logger.log(f"goal_noise_type: {self.goal_noise_type}")
        logger.log(f"infinite_tasks: {self.infinite_tasks}")
        logger.log(f"normalize_kwarg: {self.normalize_kwarg}")
        logger.log("****** Created AntDir Environment ******\n")

        super(AntDirEnv, self).__init__(task, self.num_train_tasks + self.num_eval_tasks, **kwargs)

        # Update observation space: base (27) + goal (0 or 1) + reward (0 or 1)
        obs_dim = 27 + (1 if self.goal_conditioning != "no" else 0) + (1 if self.reward_conditioning == "yes" else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
    
    def reset(self, **kwargs):
        obs = super(AntDirEnv, self).reset(**kwargs)
        return self._get_obs()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs(reward)
        return (
            ob,
            reward,
            done,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                torso_velocity=torso_velocity,
            ),
        )

    def _get_obs(self, reward=0.0):
        obs = super(AntDirEnv, self)._get_obs()

        if self.goal_conditioning == "yes":
            goal = self._goal
            if self.normalize_kwarg:
                goal = helpers.normalize_angle_to_pi_pi(goal)
            obs = np.concatenate([obs, np.array([goal])], axis=0)
        elif self.goal_conditioning == "no":
            pass
        elif self.goal_conditioning == "fixed_noise":
            goal = self._goal + self._goal_noise
            if self.normalize_kwarg:
                goal = helpers.normalize_angle_to_pi_pi(goal)
            obs = np.concatenate([obs, np.array([goal])], axis=0)
        elif self.goal_conditioning == "yes_relative":
            self_orientation = self.get_torso_orientation()
            assert self.normalize_kwarg
            relative_orientation = helpers.normalize_angle_to_pi_pi(self._goal - self_orientation)
            obs = np.concatenate([obs, np.array([relative_orientation])], axis=0)
        elif self.goal_conditioning == "yes_relative_noise":
            self_orientation = self.get_torso_orientation()
            assert self.normalize_kwarg
            relative_orientation = helpers.normalize_angle_to_pi_pi(self._goal - self_orientation + self._goal_noise)
            obs = np.concatenate([obs, np.array([relative_orientation])], axis=0)
        else:
            raise NotImplementedError(f"Unidentified goal conditioning: {self.goal_conditioning}")

        if self.reward_conditioning == "yes":
            obs = np.concatenate([obs, np.array([reward])], axis=0)
        
        return obs

    def sample_tasks(self, num_tasks):
        assert self.task_mode is not None, f"{self.task_mode}"
        assert self.num_train_tasks is not None and self.num_eval_tasks is not None, f"{self.num_train_tasks}, {self.num_eval_tasks}"
        if self.task_mode == "circle":
            n_tasks = self.num_train_tasks + self.num_eval_tasks
            angles = np.linspace(0, np.pi * 2, num=n_tasks, endpoint=False)
            assignment = np.zeros((n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, n_tasks, self.num_train_tasks, endpoint=False)).astype(int)] = True
            self.train_goals = angles[assignment]
            self.eval_goals = angles[np.logical_not(assignment)]
        elif self.task_mode == "circle_down_up":
            self.train_goals = np.linspace(np.pi, np.pi * 2, num=self.num_train_tasks, endpoint=False)
            self.eval_goals = np.linspace(0, np.pi, num=self.num_eval_tasks, endpoint=False)
        elif self.task_mode == "circle_left_right_up_down":
            train_goals_right = np.linspace(-np.pi / 4, np.pi / 4, num = self.num_train_tasks // 2, endpoint=False)
            train_goals_left = np.linspace(np.pi * 3 / 4, np.pi * 5 / 4, num = self.num_train_tasks - self.num_train_tasks // 2, endpoint=False)
            self.train_goals = np.concatenate([train_goals_right, train_goals_left], axis=0)
            eval_goals_top = np.linspace(np.pi / 4, np.pi * 3 / 4, num = self.num_eval_tasks // 2, endpoint=False)
            eval_goals_down = np.linspace(np.pi * 5 / 4, np.pi * 7 / 4, num = self.num_eval_tasks - self.num_eval_tasks // 2, endpoint=False)
            self.eval_goals = np.concatenate([eval_goals_top, eval_goals_down], axis=0)
        elif self.task_mode == "circle_down_quarter":
            self.train_goals = np.linspace(np.pi * 5 / 4, np.pi * 7 / 4, num = self.num_train_tasks, endpoint=False)
            self.eval_goals = np.linspace(-np.pi / 4, np.pi * 5 / 4, num = self.num_eval_tasks, endpoint=False)
        else:
            raise NotImplementedError(f"{self.task_mode} not allowed.")
        
        self.goals = np.concatenate([self.train_goals, self.eval_goals], axis=0)
        self.tasks = [{"goal": goal} for goal in self.goals]
        return self.tasks

    def train_task_distribution(self):
        if self.infinite_tasks != "yes":
            return self.goals[np.random.randint(len(self.goals))]
        elif self.task_mode == "circle":
            return np.random.uniform(0, 2 * np.pi)
        elif self.task_mode == "circle_down_up":
            return np.random.uniform(np.pi, 2 * np.pi)
        elif self.task_mode == "circle_left_right_up_down":
            angle = np.random.uniform(-np.pi / 4, 3 * np.pi / 4)
            if angle > np.pi / 4:
                angle += np.pi / 2
            return angle
        elif self.task_mode == "circle_down_quarter":
            return np.random.uniform(np.pi * 5 / 4, np.pi * 7 / 4)
        else:
            raise NotImplementedError(f"{self.task_mode} not allowed.")

    def _sample_raw_task(self):
        assert self.forward_backward == True
        velocity = np.random.choice([-1.0, 1.0])  # not 180 degree
        task = {"goal": velocity}
        return task
    
    def reset_task(self, goal, override_task=None):
        if goal is not None:
            assert isinstance(goal, float)
            if self.infinite_tasks != "yes":
                assert goal in self.goals
            self._goal = goal
            self._task = {"goal": goal}
        else:
            self._goal = self.train_task_distribution()
            self._task = {"goal": self._goal}

        if self.goal_noise_type == "normal":
            self._goal_noise = np.random.randn() * self.goal_noise_magnitude
        elif self.goal_noise_type == "uniform":
            self._goal_noise = np.random.uniform(-1, 1) * self.goal_noise_magnitude
        elif self.goal_noise_type == "constrained_normal":
            self._goal_noise = np.random.randn() * self.goal_noise_magnitude
            self._goal_noise = np.clip(self._goal_noise, -self.goal_noise_magnitude, self.goal_noise_magnitude)
        else:
            self._goal_noise = 0.0
        
        if override_task is not None:
            self._task = override_task
            self._goal = override_task['goal']

        self.reset()

    def render_pos(self) -> np.ndarray:
        return np.concatenate([np.array(self.get_body_com("torso"))[:2], np.array([self.get_torso_orientation()])], axis=0)

    def get_torso_orientation(self) -> float:
        """
        Get the current orientation (yaw angle) of the ant torso in 2D space.

        Returns:
            float: Yaw angle in radians, where 0 is along the positive x-axis,
                   and the angle increases counter-clockwise.
        """
        # Get quaternion from qpos: [x, y, z, qw, qx, qy, qz]
        qpos = self.data.qpos
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]

        # Convert quaternion to Euler angles and extract yaw (z-axis rotation)
        roll, pitch, yaw = helpers.quaternion_to_euler(qw, qx, qy, qz)

        return yaw

    def set_return_obs_type(self, return_obs_type):
        """Placeholder method for compatibility with VariBadWrapper."""
        pass

    def annotation(self) -> str:
        info = {
            '_goal': self._goal,
        }
        if self.goal_conditioning == "fixed_noise" or self.goal_conditioning == "yes_relative_noise":
            info['_goal_noise'] = self._goal_noise
        return str(info)
