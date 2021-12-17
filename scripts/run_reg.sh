#! /bin/bash
INPUT_IMAGES=$1
DET_FOLDER=$2
OUTPUT_FOLDER=$3

WEIGHT_PATH="weights/transformer/best.pth"
CONFIG_PATH="weights/transformer/config.yml"

PYTHONPATH=paddleocr/transformer   python paddleocr/transformer/vietocr/infer.py  \
                                --images $INPUT_IMAGES \
                                --labels $DET_FOLDER \
                                --output $OUTPUT_FOLDER \
                                --weight $WEIGHT_PATH \
                                --config $CONFIG_PATH \
                                --conf_threshold 0.1