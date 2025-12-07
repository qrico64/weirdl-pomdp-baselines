# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    for typ in ["train", "eval"]:
        for typ2 in ["return", "success"]:
            column = f"metrics/{typ2}_{typ}_total" if typ2 == "return" else f"metrics/{typ2}_rate_{typ}"
            try:
                plot_comparison(f"viz/dec6/dec6_antgoalposlinear_circle_reward_x2__{typ2}_{typ}.png", [
                    ["experiments/dec4/dec4_antgoalposlinear_circle_3layers_combined_reward_x2_3",
                    "experiments/dec5/dec5_antgoalposlinear_circle_3layers_combined_reward_x2_4",
                    "experiments/dec5/dec5_antgoalposlinear_circle_3layers_combined_reward_x2_5"],
                ], column=column, labels=[
                    "reward x2",
                ], title=f"antgoal linear circle ({typ})")
            except AssertionError as e:
                raise
            except Exception:
                pass
