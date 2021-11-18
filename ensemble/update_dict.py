# write code update word by word in a dictionary edit distance threshold
import os
from argparse import ArgumentParser
from pathlib import Path

import editdistance
import numpy as np
from tqdm.auto import tqdm


DICT_PATH = f"{os.path.dirname(__file__)}/dict.txt"
print(DICT_PATH)


def update_dict(dict_path):
    """
    update dictionary
    """
    word_dictionary = []
    with open(Path(dict_path), 'rt') as f:
        for line in f.readlines():
            words = line.split()
            words = list(map(str.upper, words))
            word_dictionary.extend(words)
    return word_dictionary


def update_word_from_dictionary(word, dictionary, threshold=0.3):
    word = word.upper()
    distances = np.array([editdistance.eval(
        word, dict_word) / len(dict_word) for dict_word in dictionary])
    min_index = np.argmin(distances)
    min_val = np.min(distances)
    if min_val < threshold:  # smaller distance is better
        return dictionary[min_index], min_val, True
    return word, min_val, False


if __name__ == "__main__":
    # example usage:
    # python update_dict.py ./data/* --threshold 0.3

    word_dictionary = update_dict(DICT_PATH)

    parser = ArgumentParser()
    parser.add_argument('txt_paths', nargs='+', type=Path)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output_dir', type=Path, default='outputs')
    args = parser.parse_args()
    txt_paths = args.txt_paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache = {}
    replace_word_counts = 0
    for txt_path in tqdm(txt_paths):
        output_path = output_dir / txt_path.name
        with open(txt_path, 'rt', encoding='utf8') as fi, open(output_path, 'wt', encoding='utf8') as fo:
            for line in fi.readlines():
                splits = line.strip().split(',')
                text = ''.join(splits[8:])
                update_stat = False
                if text in cache:
                    new_text, distance, update_stat = cache[text]
                else:
                    new_text, distance, update_stat = update_word_from_dictionary(
                        text, word_dictionary, args.threshold)
                    cache[text] = (new_text, distance, update_stat)
                if update_stat:
                    replace_word_counts += 1
                # if new_text != text:
                #     print('[{}] "{}" to "{}". Distance = {}'.format(
                #         txt_path.name, text, new_text, distance))

                out_line = ','.join(splits[:8]) + f',{new_text}\n'
                fo.write(out_line)

    print(f'Total replace words: {replace_word_counts}')
