# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    plot_comparison("viz/nov5/nov5_down_up_task_by_noise_level.png", [
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_3",],
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_2",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_3"],
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_2",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_3"],
        ["/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_3",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_3"],
    ], column="metrics/return_eval_total", labels=[
        "noiseless",
        "normal_01",
        "normal_02",
        "normal_04",
    ], title="circle_down_up 4 Tasks (Eval)")
    plot_comparison("viz/nov5/nov5_down_up_task_by_noise_level_train.png", [
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_0_3",],
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_2",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_01_3"],
        ["experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_1",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_2",
        "experiments/nov4/nov4_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_02_3"],
        ["/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_noiselessevals_antdir_circle_down_up_norm_4tasks_goal_normal_04_3",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_1",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_2",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/weirdl-pomdp-baselines/experiments/oct31/oct31_fixedtrajlength_antdir_circle_down_up_norm_4tasks_goal_normal_04_3"],
    ], column="metrics/return_train_total", labels=[
        "noiseless",
        "normal_01",
        "normal_02",
        "normal_04",
    ], title="circle_down_up 4 Tasks (Train)")
