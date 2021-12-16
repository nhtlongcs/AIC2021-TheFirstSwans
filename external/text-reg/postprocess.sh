# PYTHONPATH=. python vietocr/infer.py  \
#                 --images $1 \
#                 --labels $2 \
#                 --output $3 \
#                 --weight "weights/latest.pth" \
#                 --config "weights/config.yml" \
#                 --conf_threshold 0.1
# $1: ocr output folder 
# $2: phrase output folder
# $3: output split box folder

PYTHONPATH=. python vietocr/postprocess/split_phrase.py $1 $2 

PYTHONPATH=. python vietocr/postprocess/split_box_by_token.py  \
                --text   $2 \
                --output $3