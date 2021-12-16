import os
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import sys

GT_PATH = sys.argv[1]
PRED_PATH = sys.argv[2]


def calc_iou_polygons(polygon1, polygon2):
    # Calculate Intersection and union, and tne IOU
    polygon_intersection = polygon1.intersection(polygon2).area
    polygon_union = polygon1.union(polygon2).area
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


class MatchingPairs:
    def __init__(
        self, pred_polygons, gt_polygons, min_iou=0.5, match_text=False, eps=0.000001
    ) -> None:
        self.eps = eps
        self.pred_polygons = pred_polygons
        self.gt_polygons = gt_polygons
        self.min_iou = min_iou
        self.match_text = match_text

        self.matched = False
        self.pairs = []

    def assign_pair(self, polygon1, polygon2, iou):
        polygon1.set_matched()
        polygon2.set_matched()
        self.pairs.append([polygon1, polygon2, iou])

    def match_pairs(self):
        """
        Loop through ground truth polygons, for each check whether exists prediction match IOU threshold and text
        """
        with open("log.txt", "a", encoding="utf-8") as f:

            total = 0
            true = 0
            if len(self.pred_polygons) == 0:
                self.matched = True
                return 0, 0

            for poly in self.gt_polygons:

                # Find predicted polygon that match most
                matched, iou = poly.find_best_matching_polygon(self.pred_polygons)
                if iou >= self.min_iou and matched is not None:
                    if self.match_text:
                        total += 1
                        if poly.text != matched.text:
                            f.write(
                                f"Name: {poly.name} || GT:  {poly.text} || Pred: {matched.text}\n"
                            )
                            continue
                        true += 1

                    # If pred and gt matched completedly, check it
                    self.assign_pair(poly, matched, iou)
            self.matched = True
        return true, total

    def get_acc(self):
        true, total = self.match_pairs()
        return true, total

    def get_false_positive(self):
        if not self.matched:
            self.match_pairs()

        count = 0
        for polygon in self.pred_polygons:
            if not polygon.is_matched():
                count += 1
        return count

    def get_false_negative(self):
        if not self.matched:
            self.match_pairs()

        count = 0
        for polygon in self.gt_polygons:
            if not polygon.is_matched():
                count += 1
        return count

    def get_true_positive(self):
        if not self.matched:
            self.match_pairs()
        return len(self.pairs)

    def get_precision(self):
        TP = self.get_true_positive()
        FP = self.get_false_positive()

        return TP / (TP + FP + self.eps)

    def get_recall(self):
        TP = self.get_true_positive()
        FN = self.get_false_negative()

        return TP / (TP + FN + self.eps)

    def get_hmean(self):
        precision = self.get_precision()
        recall = self.get_recall()

        return 2 * precision * recall / (precision + recall + self.eps)


class HMean:
    def __init__(self, min_iou=0.5, match_text=True, eps=1e-6) -> None:
        self.eps = eps
        self.min_iou = min_iou
        self.match_text = match_text

    def calculate(self, pred_polygons, gt_polygons):
        TP = 0
        FP = 0
        FN = 0
        true = 0
        total = 0

        for pred_polygon, gt_polygon in zip(pred_polygons, gt_polygons):
            matched_pairs = MatchingPairs(
                pred_polygon,
                gt_polygon,
                min_iou=self.min_iou,
                match_text=self.match_text,
                eps=self.eps,
            )
            true_, total_ = matched_pairs.get_acc()
            true += true_
            total += total_
            TP += matched_pairs.get_true_positive()
            FP += matched_pairs.get_false_positive()
            FN += matched_pairs.get_false_negative()
        assert total > 0, "Total number of predictions is zero"
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        acc = true / total

        return {
            "precision": precision,
            "recall": recall,
            "hmean": 2 * precision * recall / (precision + recall),
            "acc": acc,
        }


def extract(data):
    polygons = []
    texts = []

    for line in data:
        tokens = line.split(",")
        x1, y1, x2, y2, x3, y3, x4, y4 = tokens[:8]
        x1, y1, x2, y2, x3, y3, x4, y4 = list(
            map(float, [x1, y1, x2, y2, x3, y3, x4, y4])
        )
        x1, y1, x2, y2, x3, y3, x4, y4 = list(
            map(int, [x1, y1, x2, y2, x3, y3, x4, y4])
        )
        polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        text = ",".join(tokens[8:])

        if text == "###" or text == "" or text.isspace():
            continue

        polygons.append(polygon)
        texts.append(text.upper())

    return polygons, texts


if __name__ == "__main__":
    eval_ids = [i for i in range(1001, 1101)]

    total_hmean = 0

    all_pred_polygons = []
    all_gt_polygons = []
    for id in tqdm(eval_ids):
        gt_path = os.path.join(GT_PATH, f"img_{id}.txt")
        pred_path = os.path.join(PRED_PATH, f"img_{id}.txt")

        if not os.path.exists(gt_path):
            print(f"{gt_path} not found")
            continue

        with open(gt_path, "r", encoding="utf8") as f:
            gt_data = f.read().splitlines()
            gt_coords, gt_texts = extract(gt_data)

        with open(pred_path, "r", encoding="utf8") as f:
            pred_data = f.read().splitlines()
            pred_coords, pred_texts = extract(pred_data)

        pred_polygons = [
            PolygonWithText(i, text.upper(), id)
            for i, text in zip(pred_coords, pred_texts)
        ]
        gt_polygons = [
            PolygonWithText(i, text.upper(), id) for i, text in zip(gt_coords, gt_texts)
        ]
        all_gt_polygons.append(gt_polygons)
        all_pred_polygons.append(pred_polygons)

    hmean = HMean(min_iou=0.5, match_text=True, eps=1e-6)
    result = hmean.calculate(all_pred_polygons, all_gt_polygons)
    print(result)

