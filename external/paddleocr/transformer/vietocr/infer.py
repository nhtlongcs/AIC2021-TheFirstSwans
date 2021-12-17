import sys
sys.path.append('transformer')
from vietocr.tool.config import Cfg
from vietocr.predict import Predictor
from vietocr.tool.teencode import decoder
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, help='path to image folder')
parser.add_argument('--labels', type=str, help='path to detection annotation folder')
parser.add_argument('--output', type=str, help='path to output ocr annotation folder')
parser.add_argument('--teencode', action='store_true', help='whether to use teen code')
parser.add_argument('--weight', type=str, help='path to checkpoint')
parser.add_argument('--config', type=str, help='path to config')
parser.add_argument('--beam_search', action='store_true', help='path to config')
parser.add_argument('--conf_threshold', type=float, help='ocr confidence threshold')
parser.add_argument('--no_perspective', action='store_true', help='whether to use perspective warp')

CROPPED = '/content/cropped/'

def crop_box(img, boxes, out_folder, perspective=True):
    h,w,c = img.shape
    
    for i, box in enumerate(boxes):
        box_name = os.path.join(out_folder, f"{i}.png")
        
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1,x2,x3,x4))
        min_y = max(0, min(y1,y2,y3,y4))
        max_x = min(w, max(x1,x2,x3,x4))
        max_y = min(h, max(y1,y2,y3,y4))
        
        if perspective:
            tw = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
            th = int(np.sqrt((x1-x4)**2 + (y1-y4)**2))
            pt1 = np.float32([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
            pt2 = np.float32([[0, 0],
                                [tw - 1, 0],
                                [tw - 1, th - 1],
                                [0, th - 1]])

            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            cropped = cv2.warpPerspective(img, matrix, (tw, th)) 
        else:
            cropped = img[min_y:max_y, min_x:max_x, :]
            
        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, " is missing")

    return boxes


def main(predictor, args):
    BATCH_SIZE = 1

    perspective = False if args.no_perspective else True
    image_names = os.listdir(args.images)

    for image_name in tqdm(image_names):
        image_id = image_name.split('.')[0]

        ann_path = os.path.join(args.labels, image_id+'.txt')
        with open(ann_path, 'r', encoding="utf8") as f:
            data = f.read().splitlines()
            boxes = []
            for line in data:
                tokens = line.split(',')
                x1,y1,x2,y2,x3,y3,x4,y4 = tokens[:8]
                x1,y1,x2,y2,x3,y3,x4,y4 = list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4]))
                box_score = tokens[8:]
                boxes.append(np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]))
                    
        cropped_folder = os.path.join(CROPPED, f'{image_id}')

        if not os.path.exists(cropped_folder):
            os.makedirs(cropped_folder)

        img = cv2.imread(os.path.join(args.images, image_name))

        polygons = crop_box(img, boxes, out_folder=cropped_folder, perspective=perspective)

        crop_names = [str(i)+'.png' for i in range(len(os.listdir(cropped_folder)))]

        batch = []
        batch_polygons=[]

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        output_txt_path = os.path.join(args.output, image_id+'.txt')

        with open(output_txt_path, 'w') as f:
            for idx, (crop_name, polygon) in enumerate(zip(crop_names, polygons)):
                crop_path = os.path.join(cropped_folder, crop_name)
                img = Image.open(crop_path) 
                batch.append(img)
                batch_polygons.append(polygon)
                if (idx+1) % BATCH_SIZE == 0 or idx == len(polygons)-1:

                    texts, scores = predictor.predict_batch(batch, return_prob=True)

                    for text, polygon, score in zip(texts, batch_polygons, scores):

                        if score < args.conf_threshold:
                            text = "###"
                        else:
                            if len(text) > 1 and args.teencode:
                                text = decoder(text)

                        ann_text = polygon.reshape(-1).tolist()
                        f.write(','.join([str(int(i)) for i in ann_text]) + ','+text+'\n')
                    batch = []
                    batch_polygons=[]



if __name__ == '__main__':
    args = parser.parse_args()


    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=args.beam_search

    predictor = Predictor(config)
    main(predictor, args)