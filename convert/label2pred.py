import json
import os
from glob import glob
import re
import sys 
inp_folder_name = sys.argv[1]
# inp_folder_name = './mmocr-transformerocr'
out_filename = 'text_results.json'
ls = glob(os.path.join(inp_folder_name, '*.txt'))
ls.sort()
res = []
for inp_file_name in ls:
    inp_file = open(inp_file_name, 'r')
    image_id = re.findall(r'\d+', os.path.basename(inp_file_name))
    assert len(image_id) == 1, 'inp_id is not unique'
    image_id = image_id[0]
    for line in inp_file:
        line = line.strip('\n')
        line = line.split(',')
        rec = f"{','.join(line[8:])}"
        line = list(map(float, line[:8]))
        polys = [[line[i], line[i+1]] for i in range(0, 8, 2)]
        score = 1.0
        item = {
            "image_id": image_id,
            "category_id": 1,
            "polys": polys,
            "rec": rec,
            "score": score,
        }
        res.append(item)
    inp_file.close()
with open(out_filename, 'w') as f:
    json.dump(res, f)
