import cv2

# write parser for fuse bounding box
from tqdm.auto import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from nms import fuse_methods_factory
from typing import List, Dict
import re


def dict_from_submissions(
    submission_dirs: List[Path],
) -> Dict[str, Dict[str, np.ndarray]]:
    submission_data = {}
    file_names = set()
    model_id = 0
    for submission_dir in submission_dirs:
        # key = submission_dir.name
        key = model_id
        submission_data[key] = {}
        for file in submission_dir.glob("*.txt"):
            with open(file, "r") as f:
                lines = f.readlines()
                data_arr = []
                for line in lines:
                    line = line.strip()
                    if line == "":
                        continue
                    poly = list(map(float, line.split(",")[:8]))
                    conf = float(line.split(",")[8])
                    data_arr.append(poly + [conf])
                data_arr = np.array(data_arr)  # N, 9
                submission_data[key][file.name] = data_arr
                file_names.add(file.name)
        model_id += 1
    return submission_data, file_names


def get_id(path: str):
    return re.findall(r"\d+", path)[-1]


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--input-folder", "-i", type=str, nargs="+", required=True)
    parser.add_argument("--output-folder", "-o", type=str, required=True)
    parser.add_argument(
        "--fuse-method",
        "-f",
        type=str,
        default="soft_nms",
        choices=["wbf", "nms_locality", "soft_nms"],
    )
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--conf", type=float, default=0.6)
    args = parser.parse_args()
    if args.fuse_method == "wbf":
        # fuse_method = weighted_boxes_fusion_custom
        pass
    else:
        fuse_method = fuse_methods_factory[args.fuse_method]
    submission_dirs = [Path(x) for x in args.input_folder]
    print(len(submission_dirs))
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)
    data, file_names = dict_from_submissions(submission_dirs)
    print(f"Ensembling {len(data.keys())} submission")
    # Merge detections from different submissions
    if args.fuse_method != "wbf":
        for file in tqdm(file_names):
            solutions = []
            # print(file)
            for key in data.keys():
                if file in data[key]:
                    if data[key][file].shape[0] == 0:
                        continue
                    solutions.append(data[key][file])
            if len(solutions) == 0:
                continue

            solutions = np.concatenate(solutions, axis=0)
            if args.fuse_method == "soft_nms":
                solution = fuse_method(
                    solutions, Nt_thres=args.iou, threshold=args.conf
                )
            elif args.fuse_method == "nms_locality":
                solution = fuse_method(solutions, Nt_thres=args.iou)
            output_file = output_dir / file
            with open(output_file, "w") as f:
                for line in solution:
                    f.write(",".join(map(str, line)) + "\n")
