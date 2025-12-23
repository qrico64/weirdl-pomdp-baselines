# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    for typ in ["train", "eval"]:
        for typ2 in ["return", "success"]:
            column = f"metrics/{typ2}_{typ}_total" if typ2 == "return" else f"metrics/{typ2}_rate_{typ}"
            try:
                plot_comparison(f"viz/dec23/dec23_hypothesis1_version2_circle_1_2__{typ2}_{typ}.png", [
                    ["experiments/dec21/dec21_hypothesis1_markov_circle_1_2_goal_3",
                    "experiments/dec21/dec21_hypothesis1_markov_circle_1_2_goal_4",
                    "experiments/dec21/dec21_hypothesis1_markov_circle_1_2_goal_5"],
                    ["experiments/dec21/dec21_hypothesis1_nonmarkov_circle_1_2_goal_3",
                    "experiments/dec21/dec21_hypothesis1_nonmarkov_circle_1_2_goal_4",
                    "experiments/dec21/dec21_hypothesis1_nonmarkov_circle_1_2_goal_5"],
                    ["experiments/dec21/dec21_hypothesis1_nonmarkov_circle_1_2_nogoal_3",
                    "experiments/dec21/dec21_hypothesis1_nonmarkov_circle_1_2_nogoal_5"],
                ], column=column, labels=[
                    "markov with goal",
                    "non-markov with goal and rewards",
                    "non-markov with rewards",
                ], title=f"antgoal w/ train radius 1 & test radius 2 ({typ})")
            except AssertionError as e:
                raise
            except Exception:
                pass
