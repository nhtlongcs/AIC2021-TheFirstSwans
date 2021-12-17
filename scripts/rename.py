out = "/data/submission_output/"
im_folder = "/data/test_data/"

from glob import glob
from pathlib import Path
import shutil

im_paths = sorted(glob(f"{im_folder}/*"))
lbl_paths = sorted(glob(f"{out}/*"))
for im_path, lbl_path in list(zip(im_paths, lbl_paths)):
    im_name = Path(im_path).name
    lbl_path_new = f"{out}/{im_name}.txt"
    shutil.move(lbl_path, lbl_path_new)
