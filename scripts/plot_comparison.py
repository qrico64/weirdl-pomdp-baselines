# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    for typ in ["train", "eval"]:
        plot_comparison(f"viz/nov5/nov5_down_up_task_uniform_noise{'' if typ == 'eval' else '_train'}.png", [
            ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_1",
            "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_3",],
            # ["/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
            # "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
            # "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_3",
            # "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
            # "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
            # "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_3"],
            ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_04_1",
            "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_04_2",
            "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_04_3"],
            ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_02_1",
            "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_02_2",
            "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_uniform_02_3"],
        ], column=f"metrics/return_{typ}_total", labels=[
            "noiseless",
            # "normal_04",
            "uniform_04",
            "uniform_02",
        ], title=f"circle_down_up 4 Tasks Uniform Noise ({typ})")
