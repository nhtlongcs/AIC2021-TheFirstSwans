import os
import sys
sys.path.append('transformer')
from vietocr.postprocess.correct_tone import normalize_diacritics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, help='path to ocr annotation folder')
parser.add_argument('--output', type=str, help='path to save ocr annotation folder')


SPECIALS = ' - '

def filter_specials_lr(text):
    for char in SPECIALS:
        text = text.rstrip(char).lstrip(char)
    for char in SPECIALS:
        text = text.rstrip(char).lstrip(char)
    return text

def is_phone_number_with_space(tokens):
    PHONE_SEP = ".-()/ "
    for token in tokens:
        for sep in PHONE_SEP:
            token = token.replace(sep, "")
        if not token.isnumeric():
            return False
    return True

def strip_space_get_max_length(tokens):
    max_len = 0
    max_len_id = 0
    for idx, token in enumerate(tokens):
        max_len = max(max_len, len(token))
        max_len_id = idx
    return tokens[max_len_id]

def custom_postprocess(texts):
    lst = []
    for text in texts:
        lst.append(clean_text(text))
    return lst
                   
def clean_text(text):
    if len(text) > 1: #more than 1 character
        text = filter_specials_lr(text)
        
    tokens = text.split()
    
    result = text
    if len(tokens) > 1: # contains space
       
        isphone = is_phone_number_with_space(tokens)
        if isphone:
            result = text.replace(" ", "")

    result = normalize_diacritics(result, new_style=True)
    return result

def main(args):

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        
    ann_names = os.listdir(args.labels)
    for ann_name in ann_names:
        ann_path = os.path.join(args.labels, ann_name)
        new_path = os.path.join(args.output, ann_name)
        with open(ann_path, 'r') as fi, open(new_path, 'w') as fo:
            data = fi.read().splitlines()
            for line in data:
                tokens = line.split(',')
                text = tokens[-1]
                text = clean_text(text)
                out_text = ','.join([*tokens[:-1], text])
                fo.write(out_text+'\n')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)