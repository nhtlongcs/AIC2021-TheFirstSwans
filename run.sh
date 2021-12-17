#!/bin/bash --login

echo "Running inference script"

INP_IMG_DIR=/data/test_data
OUT_IMG_DIR=/data/submission_output

echo "INP_IMG_DIR=$INP_IMG_DIR"
echo "OUT_IMG_DIR=$OUT_IMG_DIR"

echo "Running maskrcnn..."
mkdir -p /workspace/output/det/ 
bash scripts/run_det.sh $INP_IMG_DIR /workspace/output/det/ /workspace/output/det/thrs/ /workspace/output/det/sort/ 
echo "Running text recognition..."
bash scripts/run_reg.sh $INP_IMG_DIR /workspace/output/det/sort/ /workspace/output/predicted/
echo "Running postprocess..."
bash scripts/run_post.sh /workspace/output/predicted/ $OUT_IMG_DIR
echo "rename files"
python scripts/rename.py
echo "Done"

chmod a+rwx */*
chmod a+rwx /data/*/*


