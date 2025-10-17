import numpy as np
from typing import Literal

from .ant_multitask_base import MultitaskAntEnv


class AntGoalEnv(MultitaskAntEnv):
    def __init__(
        self,
        task={},
        num_train_tasks:int=3,
        num_eval_tasks:int=20,
        max_episode_steps=200,
        task_mode: Literal["circle", "circle_down_up", "circle_1_2"] = "circle",
        reward_conditioning: Literal["no", "yes"] = "no",
        goal_conditioning: Literal["no", "yes", "fixed_noise"] = "no",
        goal_noise_magnitude: float = 0,
        goal_reward_scale: float = 1,
        **kwargs
    ):
        self.task_mode = task_mode
        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.reward_conditioning = reward_conditioning
        self.goal_conditioning = goal_conditioning
        self.goal_noise_magnitude = goal_noise_magnitude
        self.goal_reward_scale = goal_reward_scale
        self.goal_radius = 0.2
        super(AntGoalEnv, self).__init__(task, self.num_train_tasks + self.num_eval_tasks, **kwargs)
        self._max_episode_steps = max_episode_steps
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.linalg.norm(xposafter[:2] - self._goal) * self.goal_reward_scale

        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                goal_forward=goal_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def sample_tasks(self, num_tasks):
        assert self.task_mode is not None, f"{self.task_mode}"
        assert self.num_train_tasks is not None and self.num_eval_tasks is not None, f"{self.num_train_tasks}, {self.num_eval_tasks}"
        if self.task_mode == "circle":
            n_tasks = self.num_train_tasks + self.num_eval_tasks
            angles = np.linspace(0, np.pi * 2, num=n_tasks, endpoint=False)
            assignment = np.zeros((n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, n_tasks, self.num_train_tasks, endpoint=False)).astype(int)] = True
            self.train_goals = np.stack([np.cos(angles[assignment]), np.sin(angles[assignment])], axis=1).tolist()
            self.eval_goals = np.stack([np.cos(angles[np.logical_not(assignment)]), np.sin(angles[np.logical_not(assignment)])], axis=1).tolist()
        elif self.task_mode == "circle_down_up":
            angles = np.linspace(np.pi, np.pi * 2, num=self.num_train_tasks, endpoint=False)
            self.train_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
            angles = np.linspace(0, np.pi, num=self.num_eval_tasks, endpoint=False)
            self.eval_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
        elif self.task_mode == "circle_1_2":
            angles = np.linspace(0, np.pi * 2, num=self.num_train_tasks, endpoint=False)
            self.train_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
            angles = np.linspace(0, np.pi * 2, num=self.num_eval_tasks, endpoint=False)
            self.eval_goals = np.stack([2 * np.cos(angles), 2 * np.sin(angles)], axis=1).tolist()
        else:
            raise NotImplementedError(f"{self.task_mode} not allowed.")
        
        self.goals = np.concatenate([self.train_goals, self.eval_goals], axis=0)
        self.tasks = [{"goal": goal} for goal in self.goals]
        return self.tasks
    
    def reset(self, **kwargs):
        obs = super(AntGoalEnv, self).reset(**kwargs)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

        if self.goal_conditioning == "yes":
            obs = np.concatenate([obs, self._goal], axis=0)
        elif self.goal_conditioning == "no":
            pass
        elif self.goal_conditioning == "fixed_noise":
            obs = np.concatenate([obs, self._goal + self._goal_noise], axis=0)
        else:
            raise NotImplementedError(f"Unidentified goal conditioning: {self.goal_conditioning}")

        if self.reward_conditioning == "yes":
            reward = self.reward_no_control()
            obs = np.concatenate([obs, np.array([reward])], axis=0)
        
        return obs
    
    def reward_no_control(self):
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.linalg.norm(xposafter[:2] - self._goal)

        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.0
        reward = goal_reward - contact_cost + survive_reward

        return reward

    def render_pos(self) -> np.ndarray:
        return np.array(self.get_body_com("torso"))[:2]
    
    def annotation(self) -> str:
        return str(list(self._goal))
