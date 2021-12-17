import sys
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize

DICT_PATH_VI = os.path.join(os.path.dirname(__file__), "dict-vi.txt")

# DICT_PATH_EN = './dict-en.txt'

word_dict = {}
MAX_VAL = np.iinfo(np.int32).max


def split_phrase_to_word(phrase, max_word_len=10):
    len_phrase = len(phrase)
    dp = np.full(len_phrase + 1, MAX_VAL)
    trade = np.zeros(len_phrase + 1, dtype="int")
    for ind in range(0, len(phrase)):
        for wlen in range(1, max_word_len + 1):
            if wlen > ind + 1:
                break
            subseq = phrase[ind - wlen + 1 : ind + 1]
            if subseq in word_dict:
                if ind - wlen == -1:
                    dp[ind] = 1
                    trade[ind] = wlen
                elif dp[ind] > dp[ind - wlen] + 1:
                    dp[ind] = dp[ind - wlen] + 1
                    trade[ind] = wlen

    if dp[len_phrase - 1] != MAX_VAL:
        ind = len_phrase - 1
        words = []
        while ind >= 0:
            wlen = trade[ind]
            w = phrase[ind - wlen + 1 : ind + 1]
            words.append(w)
            ind -= wlen
        words.reverse()
        return words
    else:
        return None


def no_accent_vietnamese(s):
    s = re.sub("[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
    s = re.sub("[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]", "A", s)
    s = re.sub("èéẹẻẽêềếệểễ", "e", s)
    s = re.sub("ÈÉẸẺẼÊỀẾỆỂỄ", "E", s)
    s = re.sub("òóọỏõôồốộổỗơờớợởỡ", "o", s)
    s = re.sub("ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ", "O", s)
    s = re.sub("ìíịỉĩ", "i", s)
    s = re.sub("ÌÍỊỈĨ", "I", s)
    s = re.sub("ùúụủũưừứựửữ", "u", s)
    s = re.sub("ƯỪỨỰỬỮÙÚỤỦŨ", "U", s)
    s = re.sub("ỳýỵỷỹ", "y", s)
    s = re.sub("ỲÝỴỶỸ", "Y", s)
    s = re.sub("Đ", "D", s)
    s = re.sub("đ", "d", s)
    return s


def read_dict(dict_path):
    nguyenam = (
        "UEOAIYÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ"
    )
    global word_dict
    f = open(dict_path, "r")
    lines = f.readlines()
    for l in lines:
        l = l.strip().upper()
        if len(l) > 1:
            for c in nguyenam:
                if c in l:
                    word_dict[l] = True
                    word_dict[no_accent_vietnamese(l)] = True
                    break
    for c in nguyenam:
        word_dict[l] = True


def process(texts):
    teencoded_exclude_dict = (
        "ÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬĐÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ"
    )
    cnt = 0
    results = []
    for text in texts:
        text = str(text)
        res = text
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            results.append("###")
            continue
        text = tokens[0]
        if len(text) < 4:
            results.append(res)
            continue
        flag = False
        if text in word_dict.keys():
            results.append(res)
            continue
        for c in text:
            if c in teencoded_exclude_dict:
                flag = True

        if flag:
            words = split_phrase_to_word(text)
            if words is not None:
                cnt += 1
                text = text + "".join(tokens[1:])
                res = " ".join(words) + "".join(tokens[1:])
                print(f"{text} -> {res}")
        results.append(res)
    # print(f"changes: {cnt} words")
    return results


if __name__ == "__main__":
    read_dict(DICT_PATH_VI)
    # df = pd.read_csv(
    #     '/home/nhtlong/workspace/scene-text/AIC-scene-text/merge-10-post.csv')
    # texts = df['text'].tolist()
    inp_folder = Path(sys.argv[1])
    out_folder = Path(sys.argv[2])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    files = inp_folder.glob("*.txt")
    for file in files:
        file_new = out_folder / file.name
        with open(file, "rt") as f, open(file_new, "wt") as g:
            texts = []
            boxes = []
            for line in f.readlines():
                # print(line)
                splits = line.strip().split(",")
                boxes.append(", ".join(splits[:8]))
                texts.append(",".join(splits[8:]))
            res = process(texts)
            for b, r in list(zip(boxes, res)):
                g.write(f"{b},{r}\n")
        f.close()
        g.close()
