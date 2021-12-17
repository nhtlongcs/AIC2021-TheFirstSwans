import os
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='path to image folder')
parser.add_argument('-o', '--output', type=str, help='path to detection annotation folder')
parser.add_argument('-t', '--threshold', type=float, default=0.2, help='iou threshold')


def calc_iou_polygons(polygon1, polygon2):
    # Calculate Intersection and union, and tne IOU
    polygon_intersection = polygon1.intersection(polygon2).area
    polygon_union = polygon1.union(polygon2).area
    if polygon_union == 0:
        iou = 0
    else:
        iou = polygon_intersection / polygon_union 
    return iou

class PolygonWithText(Polygon):
    def __init__(self, coords, text, name=None) -> None:
        super().__init__(coords)
        self.name = name
        self.polygon = coords
        self.text = text
        self.matched = False

    def get_polygon(self):
        return self.polygon

    def is_matched(self):
        return self.matched

    def set_matched(self):
        self.matched = True

    def find_best_matching_polygon(self, polygons):
        best_iou = 0
        matched_polygon = None
        for polygon in polygons:
            iou = calc_iou_polygons(self, polygon)
            if iou > best_iou:
                matched_polygon = polygon
                best_iou = iou
        return matched_polygon, best_iou

    def does_intersect_iou(self, polygons, threshold):
        for polygon in polygons:
            iou = calc_iou_polygons(self, polygon)
            if iou > threshold and iou != 1:
                return polygon
        return None

    def get_info(self):
        return self.polygon, self.text


def fuse(polys, iou_threshold):
    fused = False
    res_polys = []
    for poly in polys:
        matched_poly = poly.does_intersect_iou(polys, iou_threshold)

        if matched_poly is None:  # No intersect
            res_polys.append(poly)
        elif poly.area > matched_poly.area: # Intersect but this polygon is bigger
            res_polys.append(poly)
        else:
            fused = True                    # else

    return res_polys, fused
        
    
def extract(data):
    polygons = []
    texts = []

    for line in data:
        tokens = line.split(',')
        x1,y1,x2,y2,x3,y3,x4,y4 = tokens[:8]
        x1,y1,x2,y2,x3,y3,x4,y4 = list(map(float, [x1,y1,x2,y2,x3,y3,x4,y4]))
        x1,y1,x2,y2,x3,y3,x4,y4 = list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4]))
        polygon = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        text = ','.join(tokens[8:])

        if text == '###' or text == '' or text.isspace():
            continue

        polygons.append(polygon)
        texts.append(text.upper())

    return polygons, texts


if __name__ == '__main__':
    args = parser.parse_args()

    ann_names = os.listdir(args.input)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    all_pred_polygons = []
    all_gt_polygons = []
    for ann_name in tqdm(ann_names):
        ann_path = os.path.join(args.input, ann_name)

        with open(ann_path, 'r', encoding='utf8') as f:
            gt_data = f.read().splitlines()
            gt_coords, gt_texts = extract(gt_data)

        gt_polygons = [PolygonWithText(i, text.upper(), ann_name) for i, text in zip(gt_coords, gt_texts)]
        
        res_polygons, check = fuse(gt_polygons, args.threshold)

        if check:
            print("Fuse polygons in", ann_name)

        new_path = os.path.join(args.output, ann_name)
        with open(new_path, 'w', encoding='utf8') as fo:
            for poly in res_polygons:
                ann, text = poly.get_info()
                (x1,y1),(x2,y2),(x3,y3),(x4,y4) = ann
                x1,y1,x2,y2,x3,y3,x4,y4 = list(map(str, [x1,y1,x2,y2,x3,y3,x4,y4]))
                out_text = ','.join([x1,y1,x2,y2,x3,y3,x4,y4,text])
                fo.write(out_text+'\n')
