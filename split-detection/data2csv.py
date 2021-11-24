
import argparse
import pandas as pd
from pathlib import Path
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp',
                        type=str,
                        help='input txts folder')
    parser.add_argument('--out',
                        type=str,
                        default='data.csv',
                        help='output filename')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_id = []
    class_id = []
    polygon = []
    text = []
    args.inp = args.inp.rstrip('/')
    lbl_paths = glob(f'{args.inp}/*.txt')
    for path in lbl_paths:
        path = Path(path)
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                image_id.append(path.stem)
                class_id.append(1)
                text.append(','.join(line[8:]))
                polygon.append(line[:8])

    # Save to csv
    df = pd.DataFrame(
        {'image_id': image_id, 'class_id': class_id, 'polygon': polygon, 'text': text})

    df.to_csv(args.out, index=False)
