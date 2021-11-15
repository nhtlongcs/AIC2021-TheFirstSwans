import os
from glob import glob
import re
root = './'
inp_folder_name = os.path.join(root, 'labels')
out_folder_name = os.path.join(root, 'labels2')

os.makedirs(out_folder_name, exist_ok=True)
ls = glob(os.path.join(inp_folder_name, '*.txt'))
ls.sort()
for inp_file_name in ls:
    inp_file = open(inp_file_name, 'r')
    inp_id = re.findall(r'\d+', os.path.basename(inp_file_name))
    assert len(inp_id) == 1, 'inp_id is not unique'
    out_file_name = os.path.join(out_folder_name, f'{int(inp_id[0]):07}.txt')
    out_file = open(out_file_name, 'w')
    for line in inp_file:
        line = line.strip('\n')
        line = line.split(',')
        new_line = line[:8] + [f"####{','.join(line[8:])}"]
        new_line = ','.join(new_line)
        out_file.write(new_line + '\n')
    inp_file.close()
    out_file.close()
