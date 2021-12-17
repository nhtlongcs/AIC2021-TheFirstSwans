import re
from pathlib import Path
import sys

def merge_number_and_spaces(match: re.Match) -> str:
    keyword: str = match.group(1)
    the_rest: str = match.group(2)
    # print('keyword', keyword, 'the_rest', the_rest)
    # normalize spaces in keyword
    keyword = ''.join(keyword.split())
    the_rest = the_rest.rstrip().lstrip()
    # print(the_rest)

    number_pattern = re.compile(r'(\d[\d \.]*\d)')
    the_rest = re.sub(number_pattern, join_numbers_split_by_spaces, the_rest)
    the_rest = ' '.join(the_rest.split())

    if the_rest == '':
        return keyword

    return f'{keyword} {the_rest}'

def join_numbers_split_by_spaces(match: re.Match):
    # '  211  212  12 ' -> '211 21212'
    s = match.group(1)
    s = ''.join(str.split(s))
    return s

pattern_format = '(\\b{}\\b[\s\\:\.,]*)(.*)'

keywords = [
    # Phone
    'TEL',
    'FAX',
    'ĐTDĐ',  # This order (DTDD, DD) is important
    'DĐ',
    'SĐT',  # This order (SDT, DT) is important
    'ĐT',
    'HOTLINE',
    # Address
    'D/C',
    'Đ/C',
    'DC',
    'ĐC'
]

patterns = [re.compile(pattern_format.format(keyword)) for keyword in keywords]

INP_FOLDER = Path(sys.argv[1])
OUT_FOLDER = Path(sys.argv[2])
OUT_FOLDER.mkdir(exist_ok=True, parents=True)

for in_file in INP_FOLDER.glob('*.txt'):
    out_file = OUT_FOLDER / in_file.name
    with open(in_file, 'rt', encoding='utf-8') as fi, open(out_file, 'wt', encoding='utf-8') as fo:
        for i, line in enumerate(fi.readlines()):  # exclude header
            splits = line.strip().split(',')
            predict = ','.join(splits[8:])
            found_pattern = False
            new_predict = predict
            for pattern in patterns:
                results = pattern.findall(predict)
                if len(results) > 0:
                    results = results[0]
                    keyword, the_rest = results
                    # print(results)
                    new_predict = re.sub(pattern, merge_number_and_spaces, predict)
                    if new_predict != predict:
                        print(f'Line {i+1}: "{predict}" -> "{new_predict}"')
                    splits = splits[:8] + [new_predict]
                    break
            fo.write(','.join(splits) + '\n')
