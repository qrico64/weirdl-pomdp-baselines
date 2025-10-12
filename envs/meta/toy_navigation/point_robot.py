import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym import Env
from matplotlib.patches import Rectangle
from typing import Literal


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
        self,
        max_episode_steps=60,
        num_train_tasks:int=3,
        num_eval_tasks:int=20,
        modify_init_state_dist=True,
        on_circle_init_state=True,
        goal_conditioning: Literal["no", "yes", "fixed_noise"] = "no",
        goal_noise_magnitude: float = 0,
        reward_conditioning: Literal["no", "yes"] = "no",
        **kwargs
    ):

        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.n_tasks = self.num_train_tasks + self.num_eval_tasks
        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.modify_init_state_dist = modify_init_state_dist
        self.on_circle_init_state = on_circle_init_state
        self.goal_conditioning = goal_conditioning
        self.goal_noise_magnitude = goal_noise_magnitude
        self.reward_conditioning = reward_conditioning

        # np.random.seed(1337)
        self.goals = [[np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 0.0)] for _ in range(self.n_tasks)]

        self.reset_task(0)
        obs_dim = (2 if self.goal_conditioning == "no" else 4) + (1 if self.reward_conditioning == "yes" else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

    def reset_task(self, idx):
        """reset goal AND reset the agent"""
        if idx is not None:
            self._goal = np.array(self.goals[idx])
            self._goal_noise = np.random.normal(size=self._goal.shape) * self.goal_noise_magnitude
        self.reset()

    def set_goal(self, goal):
        self._goal = np.asarray(goal)
        self._goal_noise = np.random.normal(size=self._goal.shape) * self.goal_noise_magnitude

    def get_current_task(self):
        # for multi-task MDP
        return self._goal.copy()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1.0, 1.0, size=(2,))
        return self._get_obs()

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def _get_obs(self):
        obs: np.ndarray
        if self.goal_conditioning == "yes":
            obs = np.concatenate([np.copy(self._state), np.copy(self._goal)], axis=0)
        elif self.goal_conditioning == "no":
            obs = np.copy(self._state)
        elif self.goal_conditioning == "fixed_noise":
            obs = np.concatenate([np.copy(self._state), np.copy(self._goal + self._goal_noise)], axis=0)
        else:
            raise NotImplementedError(f"Unidentified goal conditioning: {self.goal_conditioning}")
        
        if self.reward_conditioning == "yes":
            reward = -np.linalg.norm(self._state[:2] - self._goal[:2])
            obs = np.concatenate([obs, np.array([reward])], axis=0)
        
        return obs

    def step(self, action):
        self._state = self._state + action
        reward = -(
            (
                (self._state[0] - self._goal[0]) ** 2
                + (self._state[1] - self._goal[1]) ** 2
            )
            ** 0.5
        )

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True
        else:
            done = False

        ob = self._get_obs()
        return ob, reward, done, dict()

    def reward(self, state, action=None):
        return -(
            ((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2) ** 0.5
        )

    def viewer_setup(self):
        print("no viewer")
        pass

    def render(self):
        print("current state:", self._state)


class SparsePointEnv(PointEnv):
    """
    - tasks sampled from unit half-circle
    - reward is L2 distance given only within goal radius
    NOTE that `step()` returns the dense reward because this is used during meta-training
    the algorithm should call `sparsify_rewards()` to get the sparse rewards
    """

    def __init__(
        self,
        max_episode_steps=60,
        num_train_tasks:int=3,
        num_eval_tasks:int=20,
        goal_radius=0.2,
        modify_init_state_dist=True,
        on_circle_init_state=True,
        goal_conditioning: Literal["no", "yes", "fixed_noise"] = "no",
        task_mode:Literal["circle", "circle_down_up", "circle_1_2"]="circle",
        goal_noise_magnitude: float = 0,
        **kwargs
    ):
        super().__init__(max_episode_steps, num_train_tasks, num_eval_tasks, goal_conditioning=goal_conditioning, goal_noise_magnitude=goal_noise_magnitude, **kwargs)
        self.goal_radius = goal_radius
        self.modify_init_state_dist = modify_init_state_dist
        self.on_circle_init_state = on_circle_init_state

        # np.random.seed(1337)
        self.task_mode = task_mode
        if self.task_mode == "circle":
            n_tasks = num_train_tasks + num_eval_tasks
            angles = np.linspace(0, np.pi * 2, num=n_tasks, endpoint=False)
            assignment = np.zeros((n_tasks,), dtype=bool)
            assignment[np.round(np.linspace(0, n_tasks, num_train_tasks, endpoint=False)).astype(int)] = True
            self.train_goals = np.stack([np.cos(angles[assignment]), np.sin(angles[assignment])], axis=1).tolist()
            self.eval_goals = np.stack([np.cos(angles[np.logical_not(assignment)]), np.sin(angles[np.logical_not(assignment)])], axis=1).tolist()
        elif self.task_mode == "circle_down_up":
            angles = np.linspace(np.pi, np.pi * 2, num=num_train_tasks, endpoint=False)
            self.train_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
            angles = np.linspace(0, np.pi, num=num_eval_tasks, endpoint=False)
            self.eval_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
        elif self.task_mode == "circle_1_2":
            angles = np.linspace(0, np.pi * 2, num=num_train_tasks, endpoint=False)
            self.train_goals = np.stack([np.cos(angles), np.sin(angles)], axis=1).tolist()
            angles = np.linspace(0, np.pi * 2, num=num_eval_tasks, endpoint=False)
            self.eval_goals = np.stack([2 * np.cos(angles), 2 * np.sin(angles)], axis=1).tolist()
        else:
            raise NotImplementedError(f"{self.task_mode} not allowed.")
        
        self.goals = np.concatenate([self.train_goals, self.eval_goals], axis=0)
        self.reset_task(0)

    def sparsify_rewards(self, r):
        """zero out rewards when outside the goal radius"""
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self.step_count = 0
        if self.modify_init_state_dist:  # NOTE: in varibad, it always starts from (0,0)
            self._state = np.array(
                [np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)]
            )
            if (
                not self.on_circle_init_state
            ):  # make sure initial state is not on semi-circle
                while (
                    1 - self.goal_radius
                    <= np.linalg.norm(self._state)
                    <= 1 + self.goal_radius
                ):
                    self._state = np.array(
                        [np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)]
                    )
        else:
            self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            # sparse_reward += 1 # NOTE: varibad
            sparse_reward = 1
        d.update({"sparse_reward": sparse_reward})
        # return ob, reward, done, d
        return ob, sparse_reward, done, d

    def reward(self, state, action=None):
        return self.sparsify_rewards(super().reward(state, action))

    def is_goal_state(self):
        if np.linalg.norm(self._state - self._goal) <= self.goal_radius:
            return True
        else:
            return False

    def plot_env(self):
        ax = plt.gca()
        # plot half circle and goal position
        angles = np.linspace(0, np.pi, num=100)
        x, y = np.cos(angles), np.sin(angles)
        plt.plot(x, y, color="k")
        # fix visualization
        plt.axis("scaled")
        # ax.set_xlim(-1.25, 1.25)
        ax.set_xlim(-2, 2)
        # ax.set_ylim(-0.25, 1.25)
        ax.set_ylim(-1, 2)
        plt.xticks([])
        plt.yticks([])
        circle = plt.Circle(
            (self._goal[0], self._goal[1]), radius=self.goal_radius, alpha=0.3
        )
        ax.add_artist(circle)

    def plot_behavior(self, observations, plot_env=True, **kwargs):
        # kwargs are color and label
        if plot_env:  # whether to plot circle and goal pos..(maybe already exists)
            self.plot_env()
        # label the starting point
        plt.scatter(observations[[0], 0], observations[[0], 1], marker="x", **kwargs)
        # plot trajectory
        plt.plot(observations[:, 0], observations[:, 1], **kwargs)
