import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, help='path to det+ocr annotation folder')
parser.add_argument('--output', type=str, help='path to save output folder')

args = parser.parse_args()

PRED_DIR = args.text
PRED_SPLIT_DIR = args.output

SPLIT_TOKENIZE = [' ']
RATIO_TOK2CHAR = 0.4

def split_coord(coord, start, end):
    new_coord = np.zeros((4,2))
    # Calculate vector for width and height 
    vec_width_01 = coord[1] - coord[0]
    vec_width_23 = coord[2] - coord[3]
    vec_height = coord[3] - coord[0]
    # Split to new coord
    new_coord[0] = coord[0] + vec_width_01 * start 
    new_coord[1] = coord[0] + vec_width_01 * end
    new_coord[3] = coord[3] + vec_width_23 * start
    new_coord[2] = coord[3] + vec_width_23 * end

    return new_coord

def is_number(word):
    for c in word:
        if (c == '.'):
            continue
        if (ord(c) < ord('0') or ord(c) > ord('9')):
            return False
    return True

def merge_number(wsplit):
    res = []
    for w in wsplit:
        if (len(res) > 0 and is_number(res[-1]) and is_number(w)):
            res[-1] += w
        else:
            res.append(w)
    return res



def split_word_by_token(word, coord, tok):
    wsplit = list(filter(lambda x: x != '', word.split(tok)))
    wsplit = merge_number(wsplit)
    word_len = len(word)
    n_tok = len(wsplit) - 1
    char_width = 1 / (word_len - (1 - RATIO_TOK2CHAR) * n_tok)
    tok_width = char_width * RATIO_TOK2CHAR
    coords = []
    cur_pos = 0
    for w in wsplit:
        nxt_pos = cur_pos + len(w) * char_width
        coords.append(split_coord(coord, cur_pos, nxt_pos))
        cur_pos = nxt_pos + tok_width
    return wsplit, coords



if __name__ == '__main__':
    total_bbox = 0
    pred_file_list = os.listdir(PRED_DIR)
    os.makedirs(PRED_SPLIT_DIR, exist_ok=True)

    for fname in pred_file_list:
        fpath = os.path.join(PRED_DIR, fname)
        outpath = os.path.join(PRED_SPLIT_DIR, fname)
        f = open(fpath, 'r', encoding='utf-8')
        fw = open(outpath, 'w', encoding='utf-8')
        lines = f.readlines()
        
        for l in lines:
            total_bbox += 1
            lsplit = l.split(',')
            coords = [np.array(lsplit[:8], dtype='int').reshape((4, 2))]

            words = [','.join(lsplit[8:])]
            for tok in SPLIT_TOKENIZE:
                new_coords = []
                new_words = []
                for word, coord in zip(words, coords):
                    # if (tok in word):
                    #     print(fname, word)
                    word = word.strip()
                    splitted_words, splitted_coords = split_word_by_token(word, coord, tok)
                    if tok in word:
                        for s_coord in splitted_coords:
                            plt.plot(s_coord[:,0], s_coord[:, 1])

                    new_words.extend(splitted_words)
                    new_coords.extend(splitted_coords)
                words = new_words.copy()
                coords = new_coords.copy()
            # Write split result
            for word, coord in zip(words, coords):
                coord = coord.astype('int').reshape(-1).tolist()
                result = ','.join([str(val) for val in coord])
                result = ','.join([result, word]) + '\n'
                fw.write(result)
