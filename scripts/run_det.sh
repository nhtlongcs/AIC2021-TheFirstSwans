IMG_FOLDER=$1
DET_FOLDER=$2
THRS_DET_FOLDER=$3
SORT_DET_FOLDER=$4
# FUSE_DET_FOLDER=$5

rm -rf $DET_FOLDER
rm -rf $THRS_DET_FOLDER
rm -rf $SORT_DET_FOLDER

mkdir -p $DET_FOLDER
mkdir -p $THRS_DET_FOLDER
mkdir -p $SORT_DET_FOLDER


# ls $IMG_FOLDER | head -n 5 > img_list.txt  # debug 5 images
ls $IMG_FOLDER  > img_list.txt  # debug 5 images
python mmocr_infer.py \
    $IMG_FOLDER img_list.txt \
    weights/mmocr/maskrcnn_trainall_val.py \
    weights/mmocr/best.pth \
    --out-dir $DET_FOLDER \
    --score-thr 0.5

PYTHONPATH=. python ensemble/threshold.py \
            --txt $DET_FOLDER/out_txt_dir/* \
            --output $THRS_DET_FOLDER \
            --threshold 0.6

PYTHONPATH=. python ensemble/sort_points.py $THRS_DET_FOLDER $SORT_DET_FOLDER  

# python ensemble/fuse.py -i $SORT_DET_FOLDER -o $FUSE_DET_FOLDER --iou 0.4 --conf 0.6