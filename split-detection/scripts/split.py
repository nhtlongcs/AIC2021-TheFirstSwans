import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        type=str,
                        default='../data/train_fold.csv',
                        help='path to csv file')
    parser.add_argument('--out',
                        type=str,
                        default='../data/folds/',
                        help='output directory')
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Read CSV
df = pd.read_csv(args.csv)

# Make output directory (if not existed)
if not os.path.exists(args.out):
    os.mkdir(args.out)

# Perform independently for each unique fold id
for fid in df['fold'].unique():
    fold_out_dir = f'{args.out}/{fid}'

    # Make directory if not existed
    if not os.path.exists(fold_out_dir):
        os.mkdir(fold_out_dir)

    # Save train and val CSV
    val_df = df[df['fold'] == fid]
    val_df.to_csv(f'{fold_out_dir}/{fid}_val.csv', index=False)

    train_df = df[df['fold'] != fid]
    train_df.to_csv(f'{fold_out_dir}/{fid}_train.csv', index=False)
