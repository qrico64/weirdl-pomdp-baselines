import random
import warnings
import numpy as np
import pickle
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
from itertools import product


def get_grad_norm(model):
    grad_norm = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_norm.append(p.grad.data.norm(2).item())
    if grad_norm:
        grad_norm = np.mean(grad_norm)
    else:
        grad_norm = 0.0
    return grad_norm


def vertices(N):
    """N-dimensional cube vertices -- for latent space debug
    this is 2^N binary vector"""
    return list(product((1, -1), repeat=N))


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise NotImplementedError


def env_step(env, action):
    # action: (A)
    # return: all 2D tensor shape (B=1, dim)
    action = ptu.get_numpy(action)
    if env.action_space.__class__.__name__ == "Discrete":
        action = np.argmax(action)  # one-hot to int
    next_obs, reward, done, info = env.step(action)

    # move to torch
    next_obs = ptu.from_numpy(next_obs).view(-1, next_obs.shape[0])
    reward = ptu.FloatTensor([reward]).view(-1, 1)
    done = ptu.from_numpy(np.array(done, dtype=int)).view(-1, 1)

    return next_obs, reward, done, info


def unpack_batch(batch):
    """unpack a batch and return individual elements
    - corresponds to replay_buffer object
    and add 1 dim at first dim to be concated
    """
    obs = batch["observations"][None, ...]
    actions = batch["actions"][None, ...]
    rewards = batch["rewards"][None, ...]
    next_obs = batch["next_observations"][None, ...]
    terms = batch["terminals"][None, ...]
    return obs, actions, rewards, next_obs, terms


def select_action(
    args, policy, obs, deterministic, task_sample=None, task_mean=None, task_logvar=None
):
    """
    Select action using the policy.
    """

    # augment the observation with the latent distribution
    obs = get_augmented_obs(args, obs, task_sample, task_mean, task_logvar)
    action = policy.act(obs, deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action, action_log_prob = action
    else:
        value = None
        action_log_prob = None
    action = action.to(ptu.device)
    return value, action, action_log_prob


def get_augmented_obs(args, obs, posterior_sample=None, task_mu=None, task_std=None):

    obs_augmented = obs.clone()

    if posterior_sample is None:
        sample_embeddings = False
    else:
        sample_embeddings = args.sample_embeddings

    if not args.condition_policy_on_state:
        # obs_augmented = torchkit.zeros(0,).to(device)
        obs_augmented = ptu.zeros(
            0,
        )

    if sample_embeddings and (posterior_sample is not None):
        obs_augmented = torch.cat((obs_augmented, posterior_sample), dim=1)
    elif (task_mu is not None) and (task_std is not None):
        task_mu = task_mu.reshape((-1, task_mu.shape[-1]))
        task_std = task_std.reshape((-1, task_std.shape[-1]))
        obs_augmented = torch.cat((obs_augmented, task_mu, task_std), dim=-1)

    return obs_augmented


def update_encoding(encoder, obs, action, reward, done, hidden_state):

    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    with torch.no_grad():  # size should be (batch, dim)
        task_sample, task_mean, task_logvar, hidden_state = encoder(
            actions=action.float(),
            states=obs,
            rewards=reward,
            hidden_state=hidden_state,
            return_prior=False,
        )

    return task_sample, task_mean, task_logvar, hidden_state


def seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def recompute_embeddings(
    policy_storage,
    encoder,
    sample,
    update_idx,
):
    # get the prior
    task_sample = [policy_storage.task_samples[0].detach().clone()]
    task_mean = [policy_storage.task_mu[0].detach().clone()]
    task_logvar = [policy_storage.task_logvar[0].detach().clone()]

    task_sample[0].requires_grad = True
    task_mean[0].requires_grad = True
    task_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        reset_task = policy_storage.done[i + 1]
        h = encoder.reset_hidden(h, reset_task)

        ts, tm, tl, h = encoder(
            policy_storage.actions.float()[i : i + 1],
            policy_storage.next_obs_raw[i : i + 1],
            policy_storage.rewards_raw[i : i + 1],
            h,
            sample=sample,
            return_prior=False,
        )

        task_sample.append(ts)
        task_mean.append(tm)
        task_logvar.append(tl)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.task_mu) - torch.cat(task_mean)).sum() == 0
            assert (
                torch.cat(policy_storage.task_logvar) - torch.cat(task_logvar)
            ).sum() == 0
        except AssertionError:
            warnings.warn("You are not recomputing the embeddings correctly!")
            import pdb

            pdb.set_trace()

    policy_storage.task_samples = task_sample
    policy_storage.task_mu = task_mean
    policy_storage.task_logvar = task_logvar


class FeatureExtractor(nn.Module):
    """one-layer MLP with relu
    Used for extracting features for vector-based observations/actions/rewards

    NOTE: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch.linear is a linear transformation in the LAST dimension
    with weight of size (IN, OUT)
    which means it can support the input size larger than 2-dim, in the form
    of (*, IN), and then transform into (*, OUT) with same size (*)
    e.g. In the encoder, the input is (N, B, IN) where N=seq_len.
    """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return ptu.zeros(
                0,
            )  # useful for concat


def sample_gaussian(mu, logvar, num=None):
    if num is None:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        std = torch.exp(0.5 * logvar).repeat(num, 1)
        eps = torch.randn_like(std)
        mu = mu.repeat(num, 1)
        return eps.mul(std).add_(mu)


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + ".pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)

def today_str():
    today = datetime.today()
    dateday = today.strftime("%b").lower() + str(int(today.strftime("%d")))
    return dateday


def merge_dicts(base_dict, override_dict):
    """
    Recursively merge two dictionaries, with override_dict values taking precedence.

    Args:
        base_dict: The base dictionary
        override_dict: Dictionary with values that override the base

    Returns:
        A new dictionary with merged values
    """
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_dicts(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


def ensure_cpu_tensor(tensor):
    """
    Ensure a tensor is on CPU, converting it if necessary.

    Args:
        tensor: A torch tensor that may be on CPU or CUDA

    Returns:
        The tensor on CPU
    """
    if isinstance(tensor, list):
        return [ensure_cpu_tensor(s) for s in tensor]
    if tensor.is_cuda:
        return tensor.cpu()
    return tensor


def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        qw: Quaternion scalar component (w)
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component

    Returns:
        tuple: (roll, pitch, yaw) in radians
            - roll: Rotation around x-axis
            - pitch: Rotation around y-axis
            - yaw: Rotation around z-axis
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def normalize_angle_to_pi_pi(angle: float):
    """
    Normalize an angle to the range [-pi, pi).

    Args:
        angle: Angle in radians

    Returns:
        float: Normalized angle in the range [-pi, pi)
    """
    angle = angle - np.floor(angle / (2 * np.pi)) * (2 * np.pi)
    if angle >= np.pi:
        angle -= np.pi * 2
    assert -np.pi <= angle and angle < np.pi, f"{angle}"
    return angle

