from pathlib import Path
import cv2
import numpy as np
import os

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"
TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = [
    "aˇ",
    "aˆ",
    "Aˇ",
    "Aˆ",
    "eˆ",
    "Eˆ",
    "oˆ",
    "o˒",
    "Oˆ",
    "O˒",
    "u˒",
    "U˒",
    "D^",
    "d^",
]


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()


def parse_tone(word):
    res = ""
    tone = ""
    for char in word:
        if char in dictionary:
            for group in groups:
                if char in group:
                    if tone == "":
                        tone = TONES[group.index(char)]
                    res += group[0]
        else:
            res += char
    res += tone
    return res


def full_parse(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res


def correct_tone_position(word):
    word = word[:-1]
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])

    if len(recognition) > 1:
        replace_char = correct_tone_position(recognition)
        if recognition[-1] in TONES:
            tone = recognition[-1]
            recognition = recognition[:-1]
            for group in groups:
                if replace_char in group:
                    recognition = recognition.replace(
                        replace_char, group[TONES.index(tone)]
                    )
    return recognition


def crop_box(img, boxes, out_folder, sort=True):
    h, w, c = img.shape
    if sort:
        boxes = sort_box(boxes)

    for i, box in enumerate(boxes):
        box_name = os.path.join(out_folder, f"{i}.png")

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
        x1, y1, x2, y2, x3, y3, x4, y4 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            int(x3),
            int(y3),
            int(x4),
            int(y4),
        )
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)

        min_x = max(0, min(x1, x2, x3, x4))
        min_y = max(0, min(y1, y2, y3, y4))
        max_x = min(w, max(x1, x2, x3, x4))
        max_y = min(h, max(y1, y2, y3, y4))

        cropw = abs(max_x - min_x)
        croph = abs(max_y - min_y)
        area = cropw * croph
        if area == 0:
            continue
        if croph < 32:
            cropped = img[min_y:max_y, min_x:max_x]
        else:
            tw = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            th = int(np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2))
            pt1 = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            pt2 = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            cropped = cv2.warpPerspective(img, matrix, (tw, th))

        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(x1, y1, x2, y2, x3, y3, x4, y4)
            print(min_x, min_y, max_x, max_y)
            print(img.shape)
            print(box_name, " is missing")

    return boxes


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0], box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(
        sorted_boxes,
        key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]],
    )
    return sorted_boxes


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


if __name__ == "__main__":
    import sys

    inp_folder = Path(sys.argv[1])
    out_folder = Path(sys.argv[2])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    files = inp_folder.glob("*.txt")
    for file in files:
        # image = cv2.imread(
        #     '../data/vietnamese_original/train_images/im0002.jpg')
        file_new = out_folder / file.name
        with open(file, "rt") as f, open(file_new, "wt") as g:
            boxes = []
            confs = []
            for line in f.readlines():
                splits = line.strip().split(",")
                points = list(map(int, splits[:8]))
                confident = float(splits[-1])
                # if confident <= 0.3:
                #     continue
                box = np.array(points, dtype=np.int32).reshape(-1, 2)
                boxes.append(box)
                confs.append(confident)
            boxes = sort_box(boxes)
            for box, conf in list(zip(boxes, confs)):
                box = np.array(box).astype(np.int32)
                # cv2.polylines(image, [box], True,
                #               (255, 255, 0), 2, cv2.LINE_AA)
                # for point, color in zip(box, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]):
                #     print(point)
                #     cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
                # write box to text
                g.write(",".join(map(str, box.reshape(-1))))
                g.write(f",{conf}\n")
        f.close()
        g.close()
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
