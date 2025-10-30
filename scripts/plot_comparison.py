# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    # plot_comparison("viz/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal__demonstration_eval2.png", [
    #     ["experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_04_4",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_04_5",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_04_6"],
    #     ["experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_02_1",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_02_2",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_02_3"],
    #     ["experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_08_1",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_08_2",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_normal_08_3"],
    #     ["experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_4",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_5",
    #     "experiments/oct29/oct29_nonparallel_antdir_circle_down_up_norm_4tasks_goal_6"],
    # ], column="metrics/return_eval_total", labels=[
    #     "noise_normal_04",
    #     "noise_normal_02",
    #     "noise_normal_08",
    #     "no_noise",
    # ], title="circle_down_up 4 Tasks")

    plot_comparison("viz/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal__eval2.png", [
        ["experiments/oct27/oct27_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative",
        "experiments/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative_5",
        "experiments/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative_5",],
        ["experiments/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative_normal_04_4",
        "experiments/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative_normal_04_5",
        "experiments/oct29/oct29_nonparallel_antdir_circle_down_quarter_norm_5tasks_goal_relative_normal_04_6"],
    ], column="metrics/return_eval_total", labels=[
        "noise_normal_04",
        "no_noise",
    ], title="circle_down_quarter 5 Tasks")
