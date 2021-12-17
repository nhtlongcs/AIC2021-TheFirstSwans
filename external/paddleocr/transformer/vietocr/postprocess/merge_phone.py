import os
import numpy as np
from tqdm import tqdm
import math
from shapely.geometry import Polygon
from sklearn.cluster import dbscan
from sklearn.metrics import pairwise_distances, euclidean_distances
from shapely.geometry import Point, Polygon
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='path to image folder')
parser.add_argument('-o', '--output', type=str, help='path to detection annotation folder')
parser.add_argument('--max_length_consider', type=int, default=7, help='maximum length for phone number to be consider merging, this help not to merge aldreadyfull numbers ')
parser.add_argument('--max_length_merge', type=int, default=9, help='maximum length for phone number, this help not merge multiple numbers into one')
parser.add_argument('--height_limit_ratio', type=float, default=1.5, help='height ratio to be check if two polygon are neighbors')
parser.add_argument('--width_limit_ratio', type=float, default=3.5, help='width ratio to be check if two polygon are neighbors')
args = parser.parse_args()

MAX_LENGTH_CONSIDER = args.max_length_consider
MAX_LENGTH_MERGE = args.max_length_merge
HEIGHT_LIMIT_RATIO = args.height_limit_ratio
WIDTH_LIMIT_RATIO = args.width_limit_ratio

def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def sort_box(boxes, texts):
    # Sort points in each box clockwise
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []

    # Sort order of boxes clockwise
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0],box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_texts = [text for text, _ in sorted(zip(texts, mid_points) , key=lambda x: [x[1][1], x[1][0]])]
    sorted_boxes = [box for box, _ in sorted(zip(sorted_boxes, mid_points) , key=lambda x: [x[1][1], x[1][0]])]

    return sorted_boxes, sorted_texts

def merge_phone_number_boxes(boxes, texts):

        num_merged = 0

        def get_phone_number_boxes(boxes, texts):
            """
            Only number and len < 7 
            """

            PHONE_SEP = ".-()/ "

            indexes = []
            new_boxes = []
            new_texts = []
            for idx, (box, text) in enumerate(zip(boxes, texts)):
                token = text
                for sep in PHONE_SEP:
                    token = token.replace(sep, "")
                if token.isnumeric() and len(token)<=MAX_LENGTH_CONSIDER:
                    indexes.append(idx)
                    new_boxes.append(box)
                    new_texts.append(text)
            return new_boxes, new_texts, indexes

        def merge_polygon(poly1, poly2, text1, text2):
            """
            Assume poly1 and poly2 are neighbors, only check sides
            """
            # a, b: polygon of ((x1, y1), (x2, y2), (x3, y3), (x4, y4))

            r1l2 = abs(poly1[1][0] - poly2[0][0])
            r2l1 = abs(poly1[0][0] - poly2[1][0])

            if r1l2 > r2l1:
                poly1, poly2 = poly2, poly1
                text1, text2 = text2, text1

            poly1 = Polygon(poly1)
            poly2 = Polygon(poly2)
            rect1 = poly1.exterior.coords
            rect2 = poly2.exterior.coords
            poly3 = Polygon([rect1[0], rect2[1], rect2[2], rect1[3]])

            text3 = text1 + text2
            return list(poly3.exterior.coords)[:-1], text3

        def compute_angle_between_two_vectors(vector_1, vector_2):
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            radians = np.arccos(dot_product)
            angle = math.degrees(radians)
            return angle

        def get_width_height(polygon):
            # get minimum bounding box around polygon
            box = polygon.minimum_rotated_rectangle

            # get coordinates of polygon vertices
            x, y = box.exterior.coords.xy

            # get length of bounding box edges
            edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

            # get length of polygon as the longest edge of the bounding box
            width = max(edge_length)

            # get width of polygon as the shortest edge of the bounding box
            height = min(edge_length)
            return width, height
        
        def get_polygon_distance(polygon1, polygon2):
            
            # p1 = a.reshape(-1, 2)
            # p2 = b.reshape(-1, 2)
            
            # vector_a = p1[1][0] - p1[0][0] 
            # vector_b = p2[1][0] - p2[0][0] 

            # angle = compute_angle_between_two_vectors(vector_a, vector_b)

            width1, height1 = get_width_height(polygon1)
            width2, height2 = get_width_height(polygon2)

            center_a = list(polygon1.centroid.coords)
            center_b = list(polygon2.centroid.coords)

            centroid_dist = euclidean_distances(center_a, center_b)
            if centroid_dist == 0:
                width_dist = 0
            else:
                width_dist = abs(centroid_dist - width1/2 - width2/2)
            height_dist = abs(center_a[0][1] - center_b[0][1])
            height_dist_limit = min([height1/HEIGHT_LIMIT_RATIO, height2/HEIGHT_LIMIT_RATIO])
            width_dist_limit = max([width1/WIDTH_LIMIT_RATIO, width2/WIDTH_LIMIT_RATIO])
            return width_dist, height_dist, width_dist_limit, height_dist_limit

        def compute_distance(a: np.ndarray, b: np.ndarray):
            # a, b: arrays of (x1, y1, x2, y2)
            
            box_a = Polygon(a.reshape(-1, 2))
            box_b = Polygon(b.reshape(-1, 2))

            width_dist, height_dist, width_dist_limit, height_dist_limit = get_polygon_distance(box_a, box_b)
            
            is_neighbor = False

            if box_a.intersects(box_b) and height_dist<height_dist_limit/2: # Distance between 2 polygon heights
                is_neighbor = True
                # print('neigh intersect', width_dist, width_dist_limit, height_dist, height_dist_limit)
            elif width_dist<width_dist_limit and height_dist<height_dist_limit: # Distance between 2 polygon width and heights
                # print('neigh', width_dist, width_dist_limit, height_dist, height_dist_limit)
                is_neighbor = True
            else:
                is_neighbor = False
                # print('not neigh', width_dist, width_dist_limit, height_dist, height_dist_limit)
            return 0 if is_neighbor else 1

        result_boxes = []
        result_texts = []

        # Get phone boxes and indexes in list
        phone_boxes, phone_texts, indexes = get_phone_number_boxes(boxes, texts)

        if len(phone_boxes)>1:
            # Make result boxes with no phone boxes
            for idx, (box, text) in enumerate(zip(boxes, texts)):
                if idx not in indexes:
                    result_boxes.append(box)
                    result_texts.append(text)

            # Sort phone boxes
            sorted_boxes, sorted_texts = sort_box(np.array(phone_boxes), phone_texts)
            features = np.array([np.array(box).reshape(-1) for box in sorted_boxes])
            
            distances = pairwise_distances(features, metric=compute_distance)

            centers, labels = dbscan(distances, eps=0.5, min_samples=0, metric='precomputed')
            merged_boxes = []
            merged_texts = []
            for cluster in np.unique(labels):
                item_indices = centers[labels == cluster]
                
                merged_box = sorted_boxes[item_indices[0]]
                merged_text = sorted_texts[item_indices[0]]
                for i in range(1, len(item_indices)):
                    if len(merged_text) >= MAX_LENGTH_MERGE:
                        merged_boxes.append(merged_box)
                        merged_texts.append(merged_text)
                        merged_box, merged_text = sorted_boxes[item_indices[i]], sorted_texts[item_indices[i]]
                    else:
                        merged_box, merged_text = merge_polygon(merged_box, sorted_boxes[item_indices[i]], merged_text, sorted_texts[item_indices[i]])

                merged_boxes.append(merged_box)
                merged_texts.append(merged_text)

            result_boxes.extend(merged_boxes)
            result_texts.extend(merged_texts)

            num_merged = len(texts) - len(result_texts)
        else:
            result_boxes = boxes
            result_texts = texts
            num_merged = len(texts) - len(result_texts)

        return result_boxes, result_texts, num_merged

if __name__ == '__main__':
    ann_names = os.listdir(args.input)
    total_merged = 0
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for ann_name in tqdm(ann_names):
        ann_path = os.path.join(args.input, ann_name)

        polygons = []
        texts = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
            for line in data:
                tokens = line.split(',')
                x1,y1,x2,y2,x3,y3,x4,y4 = tokens[:8]
                x1,y1,x2,y2,x3,y3,x4,y4 = list(map(float, [x1,y1,x2,y2,x3,y3,x4,y4]))
                x1,y1,x2,y2,x3,y3,x4,y4 = list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4]))

                polygon = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                text = ','.join(tokens[8:])
                if text == '###':
                    continue

                polygons.append(polygon)
                texts.append(text)

        new_polygons, new_texts, num_merged = merge_phone_number_boxes(polygons.copy(), texts.copy())

        if num_merged > 0:
            print('Merge ', ann_name)
            print("OLD: ", [i for i in texts if i not in new_texts])
            print("NEW: ", [i for i in new_texts if i not in texts])
            texts = new_texts
            polygons = new_polygons
            total_merged += num_merged
        
        new_path = os.path.join(args.output, ann_name)
        with open(new_path, 'w', encoding='utf-8') as fo:
            for poly, text in zip(polygons, texts):
                (x1,y1),(x2,y2),(x3,y3),(x4,y4) = poly
                x1,y1,x2,y2,x3,y3,x4,y4 = list(map(str, [x1,y1,x2,y2,x3,y3,x4,y4]))

                out = ','.join([x1,y1,x2,y2,x3,y3,x4,y4, text])
                fo.write(out+ '\n')
            
    print("Total number of merged phone boxes: ", total_merged)