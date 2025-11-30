# python scripts/plot_comparison.py

import os
from utils import helpers
from scripts.plot_csv_lib import plot_comparison

if __name__ == "__main__":
    for typ in ["train", "eval"]:
        plot_comparison(f"viz/nov29/nov29_circle_architecture_2{'' if typ == 'eval' else '_train'}.png", [
            ["experiments/nov28/nov28_logging_circle_3layers_separate_5",
            "experiments/nov28/nov28_logging_circle_3layers_separate_6"],
            ["experiments/nov29/nov29_logging_circle_3layers_combined_5",
            "experiments/nov29/nov29_logging_circle_3layers_combined_6"],
            ["experiments/nov29/nov29_logging_circle_3layers_combined_embed_x2_5",
            "experiments/nov29/nov29_logging_circle_3layers_combined_embed_x2_6"],
            "experiments/nov29/nov29_logging_circle_3layers_separate_src_key_padding_mask_5",
        ], column=f"metrics/return_{typ}_total", labels=[
            "separate / traditional",
            "combined1",
            "combined1, embed x2",
            "separate + padding"
        ], title=f"circle Transformer Architecture 2 ({typ})")
