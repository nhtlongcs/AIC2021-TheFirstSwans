
import argparse
import pandas as pd
from pathlib import Path
from glob import glob
import numpy as np
from shapely.geometry import Polygon


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    num_points = len(points)
    # resBoxes=np.empty([1,num_points],dtype='int32')
    resBoxes = np.empty([1, num_points], dtype="float32")
    for inp in range(0, num_points, 2):
        resBoxes[0, int(inp / 2)] = float(points[int(inp)])
        resBoxes[0, int(inp / 2 + num_points / 2)
                 ] = float(points[int(inp + 1)])
    pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
    return Polygon(pointMat)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp',
                        type=str,
                        help='input txts folder')
    parser.add_argument('--out',
                        type=str,
                        default='data.csv',
                        help='output filename')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_id = []
    class_id = []
    polygon = []
    text = []
    area = []
    args.inp = args.inp.rstrip('/')
    lbl_paths = glob(f'{args.inp}/*.txt')
    for path in lbl_paths:
        path = Path(path)
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                txt = ','.join(line[8:])
                pts = list(map(float, line[:8]))
                pts = list(map(int, pts))
                poly = polygon_from_points(pts)
                if (txt.isspace() or txt == '' or poly.area == 0):
                    continue
                polygon.append(pts)
                area.append(poly.area)
                image_id.append(path.stem)
                class_id.append(1)
                text.append(txt)

    # Save to csv
    df = pd.DataFrame(
        {'image_id': image_id, 'class_id': class_id, 'polygon': polygon, 'text': text, 'area': area})
    df.to_csv(args.out, index=False)
