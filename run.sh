#!/bin/bash --login

echo "Running inference script"

INP_IMG_DIR=/data/test_data
OUT_IMG_DIR=/data/submission_output

echo "INP_IMG_DIR=$INP_IMG_DIR"
echo "OUT_IMG_DIR=$OUT_IMG_DIR"

echo "Running maskrcnn..."
ls $INP_IMG_DIR | head -n 5 > img_list.txt  # debug 5 images
python mmocr_infer.py \
    $INP_IMG_DIR img_list.txt \
    weights/mmocr/maskrcnn_trainval.py \
    weights/mmocr/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth \
    --out-dir $OUT_IMG_DIR \
    --score-thr 0.3
