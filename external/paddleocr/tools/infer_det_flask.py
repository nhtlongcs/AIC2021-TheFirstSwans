# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle
import tldextract
import requests
import hashlib
import base64
from PIL import Image
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model, load_dygraph_params
from ppocr.utils.utility import get_image_file_list
import tools.program as program

from flask import Flask, request, render_template, redirect, url_for, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformer.vietocr.infer import crop_box, Predictor, Cfg

from tools.visualization.visualizer import Visualizer

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = './static/assets/uploads'
DETECTION_FOLDER = './static/assets/detections'
CROPPED_FOLDER = './static/assets/cropped/'
RECOGNITION_FOLDER = './static/assets/recognitions/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
app.config['RECOGNITION_FOLDER'] = RECOGNITION_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

MODEL_DET = None
MODEL_REC = None
OPS = None
POST_PROCESS = None
VISUALIZER = Visualizer()

def allowed_file_image(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_ALLOWED_EXTENSIONS


def download(url):
    ext = tldextract.extract(url)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'}
    r = requests.get(url, stream=True, headers=headers)

    # Get cache name by hashing image
    data = r.content
    ori_filename = url.split('/')[-1]
    _, ext = os.path.splitext(ori_filename)
    filename = hashlib.md5(data).hexdigest() + f'{ext}'

    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(path, "wb") as file:
        file.write(r.content)

    return filename, path

@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response


@app.route('/api/detect-text', methods=['POST'])
def api_call():
    if request.method == 'POST':
        response = {}
        if not request.json or 'url' not in request.json: 
            response['code'] = 404
            return jsonify(response)
        else:
            # get the base64 encoded string
            url = request.json['url']
            filename, filepath = download(url)

            min_conf = request.json['min_conf']
            min_iou = request.json['min_iou']

            det_results, save_det_path, save_res_path = inference_single_det(filepath, MODEL_DET, filename, OPS, POST_PROCESS)
            rec_results, save_rec_path = inference_single_rec(filepath, MODEL_REC, save_res_path)

            
            with open(save_rec_path, "rb") as f:
                res_im_bytes = f.read()        
            res_im_b64 = base64.b64encode(res_im_bytes).decode("utf8")
            response['res_image'] = res_im_b64
            response['filename'] = filename
            # response['det'] = det_results
            response['rec'] = rec_results
            response['code'] = 200
            return jsonify(response)
    else:
        return jsonify({"code": 400})


def draw_det_res(dt_boxes, config, img, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))


def setup_det():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    _ = load_dygraph_params(config, model, logger, None)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    return model, ops, post_process_class
    

def setup_rec():
    config = Cfg.load_config_from_file("/content/checkpoints/config_ocr.yml")
    config['weights'] = "/content/checkpoints/best_ocr.pth"
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    return Predictor(config)

def inference_single_det(file, model, image_name, ops, post_process_class):
    image_id, ext = os.path.splitext(os.path.basename(image_name))
    save_res_path = os.path.join(app.config['DETECTION_FOLDER'], f'{image_id}.txt')
    
    logger.info("infer_img: {}".format(file))
    with open(file, 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)

    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    images = paddle.to_tensor(images)
    preds = model(images)
    post_result = post_process_class(preds, shape_list)

    dt_boxes_json = []
    # parser boxes if post_result is dict
    if isinstance(post_result, dict):
        det_box_json = {}
        for k in post_result.keys():
            boxes = post_result[k][0]['points']
            dt_boxes_list = []
            for box in boxes:
                tmp_json = {"transcription": ""}
                tmp_json['points'] = box.tolist()
                dt_boxes_list.append(tmp_json)
            det_box_json[k] = dt_boxes_list
            # save_det_path = os.path.dirname(config['Global'][
            #     'save_res_path']) + "/det_results_{}/".format(k)
            # draw_det_res(boxes, config, src_img, file, save_det_path)
        # return det_box_json
    else:
        boxes = post_result[0]['points']
        scores = post_result[0]['scores']
        dt_boxes_json = []
        # write result
        for box, score in zip(boxes, scores):
            tmp_json = {"transcription": ""}
            tmp_json['points'] = box.tolist()
            tmp_json['score'] = round(float(score), 4)
            dt_boxes_json.append(tmp_json)
        save_det_path = os.path.join(app.config['DETECTION_FOLDER'], f'{image_id}{ext}')
        # draw_det_res(boxes, config, src_img, save_det_path)

    polygons = np.array([ann['points'] for ann in dt_boxes_json])
    scores = [ann['score'] for ann in dt_boxes_json]

    with open(save_res_path, 'w') as f:
        for idx, (polygon, score) in enumerate(zip(polygons, scores)):
            polygon = polygon.reshape(-1)
            f.write(','.join([str(int(i)) for i in polygon]) + ','+str(score)+'\n')

    return json.dumps(dt_boxes_json), save_det_path, save_res_path

def inference_single_rec(image_path, model, det_path):
    image_id, ext = os.path.splitext(os.path.basename(image_path))

    with open(det_path, 'r', encoding="utf8") as f:
        data = f.read().splitlines()
        boxes = []
        for line in data:
            tokens = line.split(',')
            x1,y1,x2,y2,x3,y3,x4,y4 = tokens[:8]
            x1,y1,x2,y2,x3,y3,x4,y4 = list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4]))
            box_score = tokens[8:]
            boxes.append(np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]))
                
    cropped_folder = os.path.join(app.config['CROPPED_FOLDER'], f'{image_id}')

    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)

    img = cv2.imread(image_path)
    VISUALIZER.set_image(img)

    polygons = crop_box(img, boxes, out_folder=cropped_folder, perspective=True)

    crop_names = [str(i)+'.png' for i in range(len(os.listdir(cropped_folder)))]


    output_txt_path = os.path.join(app.config['RECOGNITION_FOLDER'], image_id+'.txt')

    conf_threshold = 0.4

    _polygons = []
    texts = []
    return_json = []
    with open(output_txt_path, 'w') as f:
        for idx, (crop_name, polygon) in enumerate(zip(crop_names, polygons)):
            crop_path = os.path.join(cropped_folder, crop_name)
            img = Image.open(crop_path) 

            text, score = model.predict_batch([img], return_prob=True)
            text = text[0]
            score = score[0]
            if score < conf_threshold:
                continue

            ann_text = polygon.reshape(-1).tolist()

            return_json.append({
                'id': idx,
                'text': text,
                'polygon': ann_text
            })

            x1,y1,x2,y2,x3,y3,x4,y4 = ann_text
            _polygons.append([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
            texts.append(text)

            out_text = ','.join([str(int(i)) for i in ann_text]) + ','+text+'\n'
            f.write(out_text)

    VISUALIZER.draw_polygon_ocr(_polygons, texts)

    out_img = os.path.join(app.config['RECOGNITION_FOLDER'], f"{image_id}{ext}")
    cv2.imwrite(out_img, VISUALIZER.get_image())

    return json.dumps(return_json), out_img



if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    
    MODEL_DET, OPS, POST_PROCESS = setup_det()
    MODEL_REC = setup_rec()

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(DETECTION_FOLDER):
        os.makedirs(DETECTION_FOLDER, exist_ok=True)
    if not os.path.exists(CROPPED_FOLDER):
        os.makedirs(CROPPED_FOLDER, exist_ok=True)
    if not os.path.exists(RECOGNITION_FOLDER):
        os.makedirs(RECOGNITION_FOLDER, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    run_with_ngrok(app)
    app.run()