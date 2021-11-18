
# coding=utf-8
# Copyright (c) 2021 VinAI Research

from argparse import ArgumentParser
import editdistance
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np


dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('txt_paths', nargs='+', type=Path)
    parser.add_argument('--output_dir', type=Path, default='outputs')
    args = parser.parse_args()

    txt_paths = args.txt_paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for txt_path in tqdm(txt_paths):
        output_path = output_dir / txt_path.name
        with open(txt_path, 'rt', encoding='utf8') as fi, open(output_path, 'wt', encoding='utf8') as fo:
            for line in fi.readlines():
                splits = line.strip().split(',')
                text = ''.join(splits[8:])
                new_text = replace_all(text, dict_map)

                out_line = ','.join(splits[:8]) + f',{new_text}\n'
                fo.write(out_line)
