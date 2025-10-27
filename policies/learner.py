# -*- coding: future_fstrings -*-
import os, sys
import time

import math
import numpy as np
import torch
from torch.nn import functional as F
import gym
from pathlib import Path
from scripts import read_yaml
from utils import system

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger
from envs.parallel_env_manager import ParallelEnvManager


class Learner:
    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_agent(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        num_parallel_workers=1,
        **kwargs
    ):

        # initialize environment
        assert env_type in [
            "meta",
            "pomdp",
            "credit",
            "rmdp",
            "generalize",
            "atari",
        ]
        self.env_type = env_type
        self.num_parallel_workers = num_parallel_workers

        if self.env_type == "meta":  # meta tasks: using varibad wrapper
            from envs.meta.make_env import make_env, make_parallel_env

            self.train_env = make_env(
                env_name,
                max_rollouts_per_task,
                seed=self.seed,
                num_train_tasks=num_train_tasks,
                num_eval_tasks=num_eval_tasks,
                **kwargs,
            )  # oracle in kwargs

            # Create parallel environments for training if num_parallel_workers > 1
            if num_parallel_workers > 1:
                self.train_env_parallel = make_parallel_env(
                    env_name,
                    max_rollouts_per_task,
                    num_workers=num_parallel_workers,
                    seed=self.seed,
                    num_train_tasks=num_train_tasks,
                    num_eval_tasks=num_eval_tasks,
                    **kwargs,
                )
            else:
                self.train_env_parallel = None

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            if self.train_env.n_tasks is not None:
                # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
                # split to train/eval tasks
                self.train_tasks = np.arange(0, num_train_tasks)
                goals_temp = np.array(self.train_env.unwrapped.goals)
                logger.log(f"\n Train goals: {goals_temp[self.train_tasks]}\n")
                self.eval_tasks = np.arange(num_train_tasks, num_train_tasks + num_eval_tasks)
                logger.log(f"\n Eval goals: {goals_temp[self.eval_tasks]}\n")
            else:
                # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
                assert num_tasks == num_train_tasks == None
                assert (
                    num_eval_tasks > 0
                )  # to specify how many tasks to be evaluated each time
                self.train_tasks = []
                self.eval_tasks = num_eval_tasks * [None]

            # calculate what the maximum length of the trajectories is
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

        elif self.env_type in [
            "pomdp",
            "credit",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp
            import envs.credit_assign

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "atari":
            from envs.atari import create_env

            assert num_eval_tasks > 0
            self.train_env = create_env(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "rmdp":  # robust mdp task, using robust mdp wrapper
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            assert (
                num_eval_tasks > 0 and worst_percentile > 0.0 and worst_percentile < 1.0
            )
            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.worst_percentile = worst_percentile

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "generalize":
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            def check_env_class(env_name):
                if "Normal" in env_name:
                    return "R"
                if "Extreme" in env_name:
                    return "E"
                return "D"

            self.train_env_name = check_env_class(env_name)

            self.eval_envs = {}
            for env_name, num_eval_task in eval_envs.items():
                eval_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
                eval_env.seed(self.seed + 1)
                self.eval_envs[eval_env] = (
                    check_env_class(env_name),
                    num_eval_task,
                )  # several types of evaluation envs

            logger.log(self.train_env_name, self.train_env)
            logger.log(self.eval_envs)

            self.train_tasks = []
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # For non-meta environments, set train_env_parallel to None
        if not hasattr(self, 'train_env_parallel'):
            breakpoint()
            self.train_env_parallel = None

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done
        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        seq_model,
        separate: bool = True,
        image_encoder=None,
        reward_clip=False,
        **kwargs
    ):
        # initialize agent
        if seq_model == "mlp":
            agent_class = AGENT_CLASSES["Policy_MLP"]
            rnn_encoder_type = None
            assert separate == True
        elif "-mlp" in seq_model:
            agent_class = AGENT_CLASSES["Policy_RNN_MLP"]
            rnn_encoder_type = seq_model.split("-")[0]
            assert separate == True
        else:
            rnn_encoder_type = seq_model
            if separate == True:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]
            else:
                agent_class = AGENT_CLASSES["Policy_Shared_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if image_encoder is not None:  # catch, keytodoor
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape, **image_encoder
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            image_encoder_fn=image_encoder_fn,
            **kwargs,
        ).to(ptu.device)
        logger.log(self.agent)

        self.reward_clip = reward_clip  # for atari

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        use_nominals: bool = False,
        nominal_model_config_file: str = None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

        self.use_nominals = use_nominals
        self.nominal_model_config_file = nominal_model_config_file
        if self.use_nominals:
            self.nominal_model = pull_model(self.nominal_model_config_file, "latest", {})
            self.nominal_trajectories = {}
            for task in np.concatenate([self.train_tasks, self.eval_tasks], axis=0):
                task_info = self.train_env.unwrapped.tasks[task]
                rollouts = self.nominal_model.rollout_model(1, task_info, True)[0]
                self.nominal_trajectories[task] = rollouts

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        assert self.save_interval > 0
        assert self.save_interval % self.log_interval == 0
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.num_init_rollouts_pool * self.max_trajectory_len
            ):
                # Use parallel collection if available
                if not hasattr(self, 'train_env_parallel'):
                    breakpoint()
                if self.env_type == "meta" and hasattr(self, 'train_env_parallel') and self.train_env_parallel is not None:
                    self.collect_rollouts_parallel(
                        num_rollouts=min(self.num_parallel_workers,
                                        (self.num_init_rollouts_pool * self.max_trajectory_len - self._n_env_steps_total) // self.max_trajectory_len + 1),
                        random_actions=True,
                        parallel_env=self.train_env_parallel,
                    )
                else:
                    self.collect_rollouts(
                        num_rollouts=1,
                        random_actions=True,
                    )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            # Use parallel collection if available
            if self.env_type == "meta" and hasattr(self, 'train_env_parallel') and self.train_env_parallel is not None:
                env_steps = self.collect_rollouts_parallel(
                    num_rollouts=self.num_rollouts_per_iter,
                    parallel_env=self.train_env_parallel,
                )
            else:
                env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()
                self.save_ingeneral(perf)
        self.save_ingeneral(perf)

        # Clean up parallel environments
        self.cleanup()

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            if self.env_type == "meta" and self.train_env.n_tasks is not None:
                task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                obs = ptu.from_numpy(self.train_env.reset(task=task))  # reset task
            else:
                obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=False)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                ## determine terminal flag per environment
                if self.env_type == "meta" and "is_goal_state" in dir(
                    self.train_env.unwrapped
                ):
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    term = self.train_env.unwrapped.is_goal_state()
                    self._successes_in_buffer += int(term)
                elif self.env_type == "credit":  # delayed rewards
                    term = done_rollout
                else:
                    # term ignore time-out scenarios, but record early stopping
                    term = (
                        False
                        if "TimeLimit.truncated" in info
                        or steps >= self.max_trajectory_len
                        else done_rollout
                    )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def collect_rollouts_parallel(self, num_rollouts, random_actions=False, parallel_env: ParallelEnvManager = None):
        """
        Collect num_rollouts of trajectories in parallel across multiple environment workers.

        This achieves true parallelization by:
        1. Sending step commands to ALL active workers (non-blocking, returns immediately)
        2. Workers compute steps simultaneously in their separate processes
        3. Collecting results from ALL workers (blocking until all are done)

        :param num_rollouts: Total number of rollouts to collect
        :param random_actions: Whether to use policy to sample actions, or randomly sample action space
        :param parallel_env: ParallelEnvManager instance. If None, falls back to sequential collect_rollouts.
        :return: Number of environment steps collected
        """
        if parallel_env is None:
            return self.collect_rollouts(num_rollouts, random_actions)

        before_env_steps = self._n_env_steps_total
        num_workers = parallel_env.num_envs

        # Distribute rollouts across workers
        rollouts_per_worker = num_rollouts // num_workers
        extra_rollouts = num_rollouts % num_workers
        worker_rollout_counts = [rollouts_per_worker + (1 if i < extra_rollouts else 0)
                                  for i in range(num_workers)]

        # Initialize worker states
        worker_states = {}
        for worker_id in range(num_workers):
            if worker_rollout_counts[worker_id] > 0:
                worker_states[worker_id] = {
                    'remaining_rollouts': worker_rollout_counts[worker_id],
                    'active': False,
                    'steps': 0,
                    'obs': None,
                    # For memory-based agents
                    'obs_list': [],
                    'act_list': [],
                    'rew_list': [],
                    'next_obs_list': [],
                    'term_list': [],
                    'action': None,
                    'reward': None,
                    'internal_state': None,
                }

        # INITIAL RESET: Send reset commands to ALL workers (non-blocking)
        for worker_id in worker_states.keys():
            if self.env_type == "meta" and parallel_env.n_tasks is not None:
                task = self.train_tasks[np.random.randint(len(self.train_tasks))]
            else:
                task = None
            parallel_env.remotes[worker_id].send(('reset', {'task': task}))

        # Collect ALL reset responses (workers reset in parallel)
        for worker_id in worker_states.keys():
            msg_type, obs_np = parallel_env.remotes[worker_id].recv()
            assert msg_type == 'obs'

            obs = ptu.from_numpy(obs_np).reshape(1, obs_np.shape[-1])
            worker_states[worker_id]['obs'] = obs
            worker_states[worker_id]['active'] = True
            worker_states[worker_id]['steps'] = 0

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                worker_states[worker_id]['obs_list'] = []
                worker_states[worker_id]['act_list'] = []
                worker_states[worker_id]['rew_list'] = []
                worker_states[worker_id]['next_obs_list'] = []
                worker_states[worker_id]['term_list'] = []

            if self.agent_arch == AGENT_ARCHS.Memory:
                action, reward, internal_state = self.agent.get_initial_info()
                worker_states[worker_id]['action'] = action
                worker_states[worker_id]['reward'] = reward
                worker_states[worker_id]['internal_state'] = internal_state

        # Main loop: collect rollouts until all workers are done
        while any(ws['remaining_rollouts'] > 0 or ws['active'] for ws in worker_states.values()):

            # PHASE 1: Compute actions for all active workers (batched policy forward pass)
            active_workers = [wid for wid, ws in worker_states.items() if ws['active']]
            if not active_workers:
                break

            worker_actions = {}

            if random_actions:
                # Random actions: no batching needed, sample independently
                for worker_id in active_workers:
                    action = ptu.FloatTensor([parallel_env.action_space.sample()])
                    if not self.act_continuous:
                        action = F.one_hot(action.long(), num_classes=self.act_dim).float()
                    worker_actions[worker_id] = action
            else:
                # Batch policy forward pass for all active workers
                if self.agent_arch == AGENT_ARCHS.Memory:
                    # For memory-based agents, batch observations and hidden states
                    batch_obs = torch.cat([worker_states[wid]['obs'] for wid in active_workers], dim=0)
                    batch_prev_actions = torch.cat([worker_states[wid]['action'] for wid in active_workers], dim=0)
                    batch_rewards = torch.cat([worker_states[wid]['reward'] for wid in active_workers], dim=0)
                    batch_internal_states = [worker_states[wid]['internal_state'] for wid in active_workers]

                    # Batch forward pass through policy
                    (batch_actions, _, _, _), batch_new_internal_states = self.agent.act(
                        prev_internal_state=batch_internal_states,
                        prev_action=batch_prev_actions,
                        reward=batch_rewards,
                        obs=batch_obs,
                        deterministic=False,
                    )

                    # Distribute results back to workers
                    for idx, worker_id in enumerate(active_workers):
                        worker_actions[worker_id] = batch_actions[idx:idx+1]
                        worker_states[worker_id]['internal_state'] = batch_new_internal_states[idx] if isinstance(batch_new_internal_states, list) else [s[idx:idx+1] for s in batch_new_internal_states]
                else:
                    # For Markov agents, simply batch observations
                    batch_obs = torch.cat([worker_states[wid]['obs'] for wid in active_workers], dim=0)

                    # Batch forward pass through policy
                    batch_actions, _, _, _ = self.agent.act(batch_obs, deterministic=False)

                    # Distribute results back to workers
                    for idx, worker_id in enumerate(active_workers):
                        worker_actions[worker_id] = batch_actions[idx:idx+1]

            # PHASE 2: Send step commands to ALL active workers (non-blocking)
            # This is the key parallelization point: all workers start computing simultaneously
            for worker_id, action in worker_actions.items():
                action_np = ptu.get_numpy(action.squeeze(dim=0))
                if parallel_env.action_space.__class__.__name__ == "Discrete":
                    action_np = np.argmax(action_np)

                # Non-blocking send: returns immediately, worker starts step() in parallel
                parallel_env.remotes[worker_id].send(('step', {'action': action_np}))

            # PHASE 3: Collect results from ALL active workers (blocking)
            # Workers have been computing in parallel, now we collect their results
            worker_transitions = {}
            for worker_id in worker_actions.keys():
                msg_type, (next_obs_np, reward_np, done_np, info) = parallel_env.remotes[worker_id].recv()
                assert msg_type == 'transition'

                # Convert to torch tensors (matching utl.env_step format)
                next_obs = ptu.from_numpy(next_obs_np).view(-1, next_obs_np.shape[0])
                reward = ptu.FloatTensor([reward_np]).view(-1, 1)
                done = ptu.from_numpy(np.array(done_np, dtype=int)).view(-1, 1)

                worker_transitions[worker_id] = (next_obs, reward, done, info, worker_actions[worker_id])

            # PHASE 4: Process transitions (add to buffer, check for episode completion)
            workers_to_reset = []  # Track which workers need reset

            for worker_id, (next_obs, reward, done, info, action) in worker_transitions.items():
                state = worker_states[worker_id]
                obs = state['obs']

                # Clip reward if needed
                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                state['steps'] += 1

                # Determine terminal flag (same logic as original collect_rollouts)
                if self.env_type == "meta" and "is_goal_state" in info:
                    # The worker includes is_goal_state in info dict if the method exists
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    term = info["is_goal_state"]
                    self._successes_in_buffer += int(term)
                elif self.env_type == "credit":
                    term = done_rollout
                else:
                    term = (
                        False
                        if "TimeLimit.truncated" in info or state['steps'] >= self.max_trajectory_len
                        else done_rollout
                    )

                # Add transition to buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(action.squeeze(dim=0), dim=-1, keepdims=True)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # Memory-based agent
                    state['obs_list'].append(obs)
                    state['act_list'].append(action)
                    state['rew_list'].append(reward)
                    state['term_list'].append(term)
                    state['next_obs_list'].append(next_obs)

                # Update observation for next step
                state['obs'] = next_obs.clone()
                if self.agent_arch == AGENT_ARCHS.Memory:
                    state['action'] = action
                    state['reward'] = reward

                # Handle episode completion
                if done_rollout:
                    # For memory-based agents, add complete episode to buffer
                    if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                        act_buffer = torch.cat(state['act_list'], dim=0)
                        if not self.act_continuous:
                            act_buffer = torch.argmax(act_buffer, dim=-1, keepdims=True)

                        self.policy_storage.add_episode(
                            observations=ptu.get_numpy(torch.cat(state['obs_list'], dim=0)),
                            actions=ptu.get_numpy(act_buffer),
                            rewards=ptu.get_numpy(torch.cat(state['rew_list'], dim=0)),
                            terminals=np.array(state['term_list']).reshape(-1, 1),
                            next_observations=ptu.get_numpy(torch.cat(state['next_obs_list'], dim=0)),
                        )
                        print(
                            f"worker_{worker_id} steps: {state['steps']} term: {term} "
                            f"ret: {torch.cat(state['rew_list'], dim=0).sum().item():.2f}"
                        )

                    # Update statistics
                    self._n_env_steps_total += state['steps']
                    self._n_rollouts_total += 1

                    # Decrement remaining rollouts
                    state['remaining_rollouts'] -= 1

                    if state['remaining_rollouts'] > 0:
                        # Need to reset this worker for another rollout
                        workers_to_reset.append(worker_id)
                    else:
                        # This worker is completely done
                        state['active'] = False

            # PHASE 5: Reset workers that finished episodes (in parallel)
            if workers_to_reset:
                # Send reset commands to all workers that need it (non-blocking)
                for worker_id in workers_to_reset:
                    if self.env_type == "meta" and parallel_env.n_tasks is not None:
                        task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                    else:
                        task = None
                    parallel_env.remotes[worker_id].send(('reset', {'task': task}))

                # Collect reset responses (workers reset in parallel)
                for worker_id in workers_to_reset:
                    msg_type, obs_np = parallel_env.remotes[worker_id].recv()
                    assert msg_type == 'obs'

                    state = worker_states[worker_id]
                    obs = ptu.from_numpy(obs_np).reshape(1, obs_np.shape[-1])
                    state['obs'] = obs
                    state['steps'] = 0

                    # Reset storage for memory-based agents
                    if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                        state['obs_list'] = []
                        state['act_list'] = []
                        state['rew_list'] = []
                        state['next_obs_list'] = []
                        state['term_list'] = []

                    if self.agent_arch == AGENT_ARCHS.Memory:
                        action, reward, internal_state = self.agent.get_initial_info()
                        state['action'] = action
                        state['reward'] = reward
                        state['internal_state'] = internal_state

        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):

        np.random.shuffle(tasks)
        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        if self.env_type == "meta":
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps  # H
            obs_size = self.eval_env.unwrapped.observation_space.shape[
                0
            ]  # original size
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        else:  # pomdp, rmdp, generalize
            num_steps_per_episode = self.eval_env._max_episode_steps
            observations = None

        for task_idx, task in enumerate(tasks):
            step = 0

            if self.env_type == "meta" and self.eval_env.n_tasks is not None:
                obs = ptu.from_numpy(self.eval_env.reset(task=task))  # reset task
                observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info()

            episodes_infos = []
            episodes_infos_rewards = []
            for episode_idx in range(num_episodes):
                running_obss = []
                if "render_pos" in dir(self.eval_env.unwrapped):
                    running_obss.append(list(np.squeeze(self.eval_env.unwrapped.render_pos())))
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )
                    if "render_pos" in dir(self.eval_env.unwrapped):
                        running_obss.append(list(np.squeeze(self.eval_env.unwrapped.render_pos())))

                    # add raw reward
                    running_reward += reward.item()
                    # clip reward if necessary for policy inputs
                    if self.reward_clip and self.env_type == "atari":
                        reward = torch.tanh(reward)

                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    if self.env_type == "meta":
                        observations[task_idx, step, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    if (
                        self.env_type == "meta"
                        and "is_goal_state" in dir(self.eval_env.unwrapped)
                        and self.eval_env.unwrapped.is_goal_state()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    elif (
                        self.env_type == "generalize"
                        and self.eval_env.unwrapped.is_success()
                    ):
                        success_rate[task_idx] = 1.0  # ever once reach
                    elif "success" in info and info["success"] == True:  # keytodoor
                        success_rate[task_idx] = 1.0

                    if done_rollout:
                        # for all env types, same
                        break
                    if self.env_type == "meta" and info["done_mdp"] == True:
                        # for early stopping meta episode like Ant-Dir
                        break

                returns_per_episode[task_idx, episode_idx] = running_reward
                episodes_infos.append(running_obss)
                episodes_infos_rewards.append(running_reward)
            total_steps[task_idx] = step
            if "annotation" in dir(self.eval_env.unwrapped):
                logger.log(f"\nTask {task} ({task_idx}) ({self.eval_env.unwrapped.annotation()}):")
            logger.log(f"{episodes_infos}\n")
            logger.log(f"{episodes_infos_rewards}\n")
        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        if self.env_type == "meta":
            if self.train_env.n_tasks is not None:
                (
                    returns_train,
                    success_rate_train,
                    observations,
                    total_steps_train,
                ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                observations_eval,
                total_steps_eval,
            ) = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    observations_eval_sto,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            if self.train_env.n_tasks is not None and "plot_behavior" in dir(
                self.eval_env.unwrapped
            ):
                # plot goal-reaching trajs
                for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)  # must have task argument
                    logger.add_figure(
                        "trajectory/train_task_{}".format(i),
                        utl_eval.plot_rollouts(observations[i, :], self.eval_env),
                    )

                for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)
                    logger.add_figure(
                        "trajectory/eval_task_{}".format(i),
                        utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env),
                    )
                    if self.eval_stochastic:
                        logger.add_figure(
                            "trajectory/eval_task_{}_sto".format(i),
                            utl_eval.plot_rollouts(
                                observations_eval_sto[i, :], self.eval_env
                            ),
                        )

            if "is_goal_state" in dir(
                self.eval_env.unwrapped
            ):  # goal-reaching success rates
                # some metrics
                logger.record_tabular(
                    "metrics/successes_in_buffer",
                    self._successes_in_buffer / self._n_env_steps_total,
                )
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/success_rate_train", np.mean(success_rate_train)
                    )
                logger.record_tabular(
                    "metrics/success_rate_eval", np.mean(success_rate_eval)
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                    )

            for episode_idx in range(self.max_rollouts_per_task):
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/return_train_episode_{}".format(episode_idx + 1),
                        np.mean(returns_train[:, episode_idx]),
                    )
                logger.record_tabular(
                    "metrics/return_eval_episode_{}".format(episode_idx + 1),
                    np.mean(returns_eval[:, episode_idx]),
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                        np.mean(returns_eval_sto[:, episode_idx]),
                    )

            if self.train_env.n_tasks is not None:
                logger.record_tabular(
                    "metrics/total_steps_train", np.mean(total_steps_train)
                )
                logger.record_tabular(
                    "metrics/return_train_total",
                    np.mean(np.sum(returns_train, axis=-1)),
                )
            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )

        elif self.env_type == "generalize":
            returns_eval, success_rate_eval, total_steps_eval = {}, {}, {}
            for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
                self.eval_env = env  # assign eval_env, not train_env
                for suffix, deterministic in zip(["", "_sto"], [True, False]):
                    if deterministic == False and self.eval_stochastic == False:
                        continue
                    return_eval, success_eval, _, total_step_eval = self.evaluate(
                        eval_num_episodes_per_task * [None],
                        deterministic=deterministic,
                    )
                    returns_eval[
                        self.train_env_name + env_name + suffix
                    ] = return_eval.squeeze(-1)
                    success_rate_eval[
                        self.train_env_name + env_name + suffix
                    ] = success_eval
                    total_steps_eval[
                        self.train_env_name + env_name + suffix
                    ] = total_step_eval

            for k, v in returns_eval.items():
                logger.record_tabular(f"metrics/return_eval_{k}", np.mean(v))
            for k, v in success_rate_eval.items():
                logger.record_tabular(f"metrics/succ_eval_{k}", np.mean(v))
            for k, v in total_steps_eval.items():
                logger.record_tabular(f"metrics/total_steps_eval_{k}", np.mean(v))

        elif self.env_type == "rmdp":
            returns_eval, _, _, total_steps_eval = self.evaluate(self.eval_tasks)
            returns_eval = returns_eval.squeeze(-1)
            # np.quantile is introduced in np v1.15, so we have to use np.percentile
            cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            worst_indices = np.where(
                returns_eval <= cutoff
            )  # must be "<=" to avoid empty set
            returns_eval_worst, total_steps_eval_worst = (
                returns_eval[worst_indices],
                total_steps_eval[worst_indices],
            )

            logger.record_tabular("metrics/return_eval_avg", returns_eval.mean())
            logger.record_tabular(
                "metrics/return_eval_worst", returns_eval_worst.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_avg", total_steps_eval.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
            )

        elif self.env_type in ["pomdp", "credit", "atari"]:
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(
                self.eval_tasks
            )
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )

            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                )

        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        if self.env_type == "generalize":
            return sum([v.mean() for v in success_rate_eval.values()]) / len(
                success_rate_eval
            )
        else:
            return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)
        return save_path

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)

    def cleanup(self):
        """Clean up resources, especially parallel environments."""
        if hasattr(self, 'train_env_parallel') and self.train_env_parallel is not None:
            logger.log("Closing parallel environments...")
            self.train_env_parallel.close()
            logger.log("Parallel environments closed.")
    
    def save_ingeneral(self, log_perf):
        current_num_iters = self._n_env_steps_total // (self.num_rollouts_per_iter * self.max_trajectory_len)
        logger.log("\n****** Saving Model ******")
        logger.log(f"_n_env_steps_total: {self._n_env_steps_total}")
        logger.log(f"num_rollouts_per_iter: {self.num_rollouts_per_iter}")
        logger.log(f"max_trajectory_len: {self.max_trajectory_len}")
        logger.log(f"current_num_iters: {current_num_iters}")
        logger.log(f"log_interval: {self.log_interval}")
        logger.log(f"n_env_steps_total: {self.n_env_steps_total}")
        logger.log(f"save_interval: {self.save_interval}")

        save_dir = os.path.join(logger.get_dir(), "save")

        # Save agent model
        agent_save_path = os.path.join(save_dir, f"agent_{current_num_iters}_perf{log_perf:.3f}.pt")
        torch.save(self.agent.state_dict(), agent_save_path)
        logger.log(f"agent_save_path: {agent_save_path}")

        # Save optimizer(s) - generalizable to different agent types
        optimizer_states = {}
        for attr_name in dir(self.agent):
            attr = getattr(self.agent, attr_name)
            # Check if attribute is an optimizer (has state_dict and step methods)
            if hasattr(attr, 'state_dict') and hasattr(attr, 'step') and 'optim' in attr_name.lower():
                optimizer_states[attr_name] = attr.state_dict()

        if optimizer_states:
            optimizer_save_path = os.path.join(save_dir, f"optimizer_{current_num_iters}_perf{log_perf:.3f}.pt")
            torch.save(optimizer_states, optimizer_save_path)
            logger.log(f"optimizer_save_path: {optimizer_save_path}")
            logger.log(f"saved optimizers: {list(optimizer_states.keys())}")

        # Save learner state
        learner_state = {
            # Training progress counters
            'n_env_steps_total': self._n_env_steps_total,
            'n_env_steps_total_last': self._n_env_steps_total_last,
            'n_rl_update_steps_total': self._n_rl_update_steps_total,
            'n_rollouts_total': self._n_rollouts_total,
            'successes_in_buffer': self._successes_in_buffer,
            # Timing information
            'start_time': self._start_time,
            'start_time_last': self._start_time_last,
            # RNG states for reproducibility
            'random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        learner_save_path = os.path.join(save_dir, f"learner_{current_num_iters}_perf{log_perf:.3f}.pt")
        torch.save(learner_state, learner_save_path)
        logger.log(f"learner_save_path: {learner_save_path}")

        buffer_save_path = os.path.join(save_dir, f"buffer_{current_num_iters}_perf{log_perf:.3f}.pt")
        torch.save(self.policy_storage, buffer_save_path)
        logger.log(f"policy_storage size: {self.policy_storage.size()}")
        logger.log(f"buffer_save_path: {buffer_save_path}")

        logger.log("****** Saved Model ******\n")

        # Clean up old checkpoints: keep checkpoints divisible by save_interval OR last 3
        logger.log("****** Cleaning up old checkpoints ******")

        # Scan save directory for all checkpoint files
        for filename in os.listdir(save_dir):
            # Match files like agent_123_perf*.pt, optimizer_123_perf*.pt, learner_123_perf*.pt
            if not filename.startswith(('agent_', 'optimizer_', 'learner_', 'buffer_')):
                continue
            filepath = os.path.join(save_dir, filename)
            try:
                iter_num = int(filename.split('_')[1])
                if (iter_num % self.save_interval != 0 and current_num_iters - iter_num > self.log_interval * 2) or \
                   (filename.startswith('buffer_') and current_num_iters != iter_num):
                    logger.log(f"Removing checkpoint: {filepath}")
                    os.remove(filepath)
                    logger.log(f"Removed checkpoint: {filepath}")
            except ValueError:
                continue
            except Exception as e:
                logger.log(f"Failed to remove {filepath}: {e}")

        logger.log("****** Cleaned up old checkpoints ******\n")

    def load_ingeneral(self, agent_file: str):
        try:
            agent_file: Path = Path(agent_file)
            assert agent_file.exists()
            assert agent_file.name.startswith("agent_")

            # Construct paths to related checkpoint files
            learner_file: Path = agent_file.parent / ("learner_" + agent_file.name[6:])
            assert learner_file.exists()
            optimizer_file: Path = agent_file.parent / ("optimizer_" + agent_file.name[6:])
            assert optimizer_file.exists()

            # Find the latest buffer file (only one is kept, may not match exact iteration)
            buffer_files = [fp for fp in agent_file.parent.iterdir() if fp.is_file() and fp.name.startswith("buffer_")]
            assert len(buffer_files) > 0, "No buffer file found"
            # Sort by modification time to get the most recent one
            buffer_file: Path = max(buffer_files, key=lambda p: p.stat().st_mtime)

            logger.log("\n****** Loading Model ******")
            logger.log(f"agent_file: {agent_file}")
            logger.log(f"learner_file: {learner_file}")
            logger.log(f"optimizer_file: {optimizer_file}")
            logger.log(f"buffer_file: {buffer_file}")

            # Load agent model
            self.agent.load_state_dict(torch.load(agent_file, map_location=ptu.device))
            logger.log("Loaded agent model")

            # Load optimizer(s)
            optimizer_states = torch.load(optimizer_file, map_location=ptu.device)
            for attr_name, state_dict in optimizer_states.items():
                if hasattr(self.agent, attr_name):
                    optimizer = getattr(self.agent, attr_name)
                    optimizer.load_state_dict(state_dict)
                    logger.log(f"Loaded optimizer: {attr_name}")
            logger.log(f"Loaded optimizers: {list(optimizer_states.keys())}")

            # Load learner state
            learner_state = torch.load(learner_file, map_location=ptu.device)
            self._n_env_steps_total = learner_state['n_env_steps_total']
            self._n_env_steps_total_last = learner_state['n_env_steps_total_last']
            self._n_rl_update_steps_total = learner_state['n_rl_update_steps_total']
            self._n_rollouts_total = learner_state['n_rollouts_total']
            self._successes_in_buffer = learner_state['successes_in_buffer']
            self._start_time = learner_state['start_time']
            self._start_time_last = learner_state['start_time_last']
            logger.log("Loaded learner state")
            logger.log(f"  n_env_steps_total: {self._n_env_steps_total}")
            logger.log(f"  n_rl_update_steps_total: {self._n_rl_update_steps_total}")
            logger.log(f"  n_rollouts_total: {self._n_rollouts_total}")

            # Restore RNG states for reproducibility
            np.random.set_state(learner_state['random_state'])
            # Ensure torch_rng_state is on CPU before setting it
            torch.set_rng_state(utl.ensure_cpu_tensor(learner_state['torch_rng_state']))
            if torch.cuda.is_available() and learner_state['torch_cuda_rng_state'] is not None:
                torch.cuda.set_rng_state_all(utl.ensure_cpu_tensor(learner_state['torch_cuda_rng_state']))
            logger.log("Restored RNG states")

            # Load buffer
            self.policy_storage = torch.load(buffer_file, map_location=ptu.device)
            logger.log(f"Loaded buffer with size: {self.policy_storage.size()}")

            logger.log("****** Successfully Loaded Model ******\n")

        except Exception as e:
            logger.log(f"Failed to load from {agent_file.resolve()}: {e}")
            raise
    
    def rollout_model(self, num_episodes: int, task_dict: dict, deterministic=False):
        results = []

        if self.env_type == "meta":
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps
        else:
            num_steps_per_episode = self.eval_env._max_episode_steps

        while len(results) < num_episodes:
            current = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'next_observations': [],
            }

            if self.env_type == "meta" and self.eval_env.n_tasks is not None:
                obs = ptu.from_numpy(self.eval_env.reset(task=0, override_task=task_dict))
            else:
                obs = ptu.from_numpy(self.eval_env.reset())
            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                action, reward, internal_state = self.agent.get_initial_info()

            for _ in range(num_steps_per_episode):
                if self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=deterministic,
                    )
                else:
                    action, _, _, _ = self.agent.act(
                        obs, deterministic=deterministic
                    )

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )

                # clip reward if necessary for policy inputs
                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                current['observations'].append(obs)
                current['actions'].append(action.squeeze(dim=0))
                current['rewards'].append(reward)
                current['dones'].append(done)
                current['next_observations'].append(next_obs)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: obs <- next_obs
                obs = next_obs.clone()

                if done_rollout:
                    # for all env types, same
                    break
                if self.env_type == "meta" and info["done_mdp"] == True:
                    # for early stopping meta episode like Ant-Dir
                    break

            results.append(current)
        
        return results

def pull_model(config_file: str, checkpoint_num: str, override_args: dict):
    assert Path(config_file).name.startswith("variant_"), f"Unable to load incorrect name: {config_file}"
    print(f"Pulling from {config_file} :")
    v = read_yaml.read_yaml_to_dict(config_file)
    seed = v["seed"]
    system.reproduce(seed)
    learner = Learner(
        env_args=utl.merge_dicts(v["env"], override_args.get("env", {})),
        train_args=utl.merge_dicts(v["train"], override_args.get("train", {})),
        eval_args=utl.merge_dicts(v["eval"], override_args.get("eval", {})),
        policy_args=utl.merge_dicts(v["policy"], override_args.get("policy", {})),
        seed=seed,
    )
    
    agent_files = [fp for fp in (Path(config_file).parent / "save").iterdir() if fp.is_file() and fp.name.startswith("agent_")]
    if checkpoint_num == "latest":
        last_agent_file = max(agent_files, key=lambda fp: int(fp.name.split('_')[1]))
    elif checkpoint_num == "bestperf":
        last_agent_file = max(agent_files, key=lambda fp: float(fp.name.split('perf')[1][:-3]))
    else:
        agent_files_exact = [fp for fp in agent_files if fp.name.startswith(f"agent_{checkpoint_num}_")]
        assert len(agent_files_exact) > 0, f"{agent_files_exact}"
        last_agent_file = agent_files_exact[0]
    
    learner.load_ingeneral(last_agent_file)

    return learner



