import sys
import numpy as np
import torch
from ruamel.yaml import YAML
from absl import flags
import os
import matplotlib.pyplot as plt

from utils import system
import torchkit.pytorch_utils as ptu
from envs.meta.make_env import make_env
from envs.meta.wrappers import VariBadWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string("cfg", None, "path to configuration file")
flags.DEFINE_string("j", None, "Job name.")
flags.mark_flag_as_required("cfg")
flags.mark_flag_as_required("j")


def data_collection_cycle(
        # env: VariBadWrapper,
        # num_rollouts: int,
        # save_file_header: str,
    ):
    # buf_size = env.horizon_bamdp * num_rollouts
    # policy_storage = buffer_class(
    #     max_replay_buffer_size=buf_size,
    #     observation_dim=self.obs_dim,
    #     action_dim=self.act_dim if self.act_continuous else 1,  # save memory
    #     sampled_seq_len=sampled_seq_len,
    #     sample_weight_baseline=sample_weight_baseline,
    #     observation_type=self.train_env.observation_space.dtype,
    # )

    # for ni in range(num_rollouts):
    #     env.reset()
    
    pass


def main():
    # FLAGS(sys.argv)

    # # Load configuration
    # yaml = YAML()
    # v = yaml.load(open(FLAGS.cfg))

    # # System setup: seed
    # seed = v["seed"]
    # system.reproduce(seed)

    # torch.set_num_threads(1)
    # np.set_printoptions(precision=3, suppress=True)
    # torch.set_printoptions(precision=3, sci_mode=False)
    # ptu.set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

    # # Extract environment configuration
    # env_args = v["env"]
    # env = make_env(
    #     env_id=env_args["env_name"],
    #     episodes_per_task=1,
    #     seed=seed,
    #     num_train_tasks=env_args.get("num_train_tasks", None),
    #     num_eval_tasks=env_args.get("num_eval_tasks", None),
    # )

    # data_collection_cycle(env)

    frames = []
    for i in range(100):
        frame = np.zeros((400, 400), dtype=np.uint8)
        frame[i:i+50, i:i+50] = 255
        frames.append(frame)

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()
    fig.canvas.draw()
    fig.canvas.flush_events()

    for frame in frames:
        im.set_data(frame)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

    # keep window open when script finishes
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
