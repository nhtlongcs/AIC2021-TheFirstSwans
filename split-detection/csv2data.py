from tqdm.auto import tqdm
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp',
                        type=str,
                        help='input csv file')
    parser.add_argument('--out',
                        type=str,
                        default='labels',
                        help='output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_id = []
    class_id = []
    polygon = []
    text = []
    area = []
    args.inp = Path(args.inp)
    args.out = Path(args.out)
    args.out.mkdir(exist_ok=True)
    if args.inp.suffix == '.csv':
        ls = [args.inp]
    else:
        ls = args.inp.glob('*.csv')
    for p in tqdm(ls):
        df = pd.read_csv(p)
        folder_out = args.out / p.stem
        folder_out.mkdir(exist_ok=True, parents=True)
        image_ids = df['image_id'].unique()

        for image_id in tqdm(image_ids):
            df_image = df[df['image_id'] == image_id]
            with open(folder_out / f'{image_id}.txt', 'w') as f:
                for i in range(len(df_image)):
                    row = df_image.iloc[i]
                    line = f"{','.join(row['polygon'])}, {df_image.iloc[i]['text']}\n"
                    f.write(line)
