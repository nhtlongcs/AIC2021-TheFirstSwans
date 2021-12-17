# Variables
DET_WEIGHT=$1
DET_CONFIG=$2

REC_WEIGHT=$3
REC_CONFIG=$4

IMG_FOLDER=$5
DET_RESULTS=$6
REC_RESULTS=$7


PYTHONPATH=. python tools/infer_det.py \
    -c $DET_CONFIG \
    -o Global.checkpoints=$DET_WEIGHT \
    Eval.loader.num_workers=0 \
    PostProcess.box_thresh=0.6 \
    PostProcess.unclip_ratio=1.5 \
    PostProcess.thresh=0.3 \
    Global.infer_img=$IMG_FOLDER \
    Global.save_res_path=$DET_RESULTS \

PYTHONPATH=. python transformer/vietocr/infer.py  \
    --images $IMG_FOLDER \
    --labels "$DET_RESULTS/text_results" \
    --output "$REC_RESULTS" \
    --weight $REC_WEIGHT \
    --config $REC_CONFIG \
    --conf_threshold 0.3

PYTHONPATH=. python transformer/vietocr/postprocess/split_phrase.py \
                     "$REC_RESULTS" \
                     "$REC_RESULTS-phrase"

PYTHONPATH=. python transformer/vietocr/postprocess/split_box_by_token.py  \
            --text   "$REC_RESULTS-phrase" \
            --output "$REC_RESULTS-split"

PYTHONPATH=. python transformer/vietocr/postprocess/merge_phone.py  \
            -i       "$REC_RESULTS-split" \
            -o       "$REC_RESULTS-phone"         

PYTHONPATH=. python transformer/vietocr/postprocess/update_dict.py \
                     --txt-paths $REC_RESULTS-phone/* \
                     --threshold 0.1 \
                     --output-dir "$REC_RESULTS-dict"

PYTHONPATH=. python transformer/vietocr/postprocess/clean.py  \
            --labels       "$REC_RESULTS-dict" \
            --output       "$REC_RESULTS-final"   


