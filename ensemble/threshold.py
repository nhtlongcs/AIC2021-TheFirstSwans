# write code update word by word in a dictionary edit distance threshold
import os
from argparse import ArgumentParser
from pathlib import Path

import editdistance
import numpy as np
from tqdm.auto import tqdm


if __name__ == "__main__":
    # example usage:
    # python threshold.py --txt ./data/* --output --threshold 0.3

    parser = ArgumentParser()
    parser.add_argument("--txt-paths", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default="outputs")
    args = parser.parse_args()
    txt_paths = args.txt_paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache = {}
    removed_lines = 0
    for txt_path in tqdm(txt_paths):
        output_path = output_dir / txt_path.name
        with open(txt_path, "rt", encoding="utf8") as fi, open(
            output_path, "wt", encoding="utf8"
        ) as fo:
            for line in fi.readlines():
                splits = line.strip().split(",")
                score = float(splits[8])
                if score > args.threshold:
                    out_line = ",".join(splits[:8]) + f",{score}\n"
                    fo.write(out_line)
                else:
                    removed_lines += 1

    print(f"Total removed lines: {removed_lines}")
