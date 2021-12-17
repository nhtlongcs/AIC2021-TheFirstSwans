%cd /content/vietocr/
PYTHONPATH=. python vietocr/infer.py  --images "/content/data/TestA" \
                  --labels "/content/predicted" \
                  --output "/content/results/maskrcnn-fold" \
                  --weight "/content/drive/MyDrive/AI/Weights/vn-scene-text/transformerocr/new_weights/transformer_final_32x512_keepwidth/best.pth" \
                  --config "/content/drive/MyDrive/AI/Weights/vn-scene-text/transformerocr/new_weights/transformer_final_32x512_keepwidth/config.yml" \
                  --conf_threshold 0.3 \

PYTHONPATH=. python vietocr/postprocess/split_phrase_to_word.py \
                     "/content/content/results/pseudo" \
                     "/content/content/results/pseudo-phrase1"

PYTHONPATH=. python vietocr/postprocess/split_phrase.py \
                     "/content/content/results/pseudo-phrase1" \
                     "/content/content/results/pseudo-phrase2"

PYTHONPATH=. python vietocr/postprocess/split_box_by_token.py  \
            --text   "/content/content/results/pseudo-phrase2" \
            --output "/content/content/results/pseudo-split"

PYTHONPATH=. python vietocr/postprocess/fuse.py  \
              -i  "/content/content/results/pseudo-split"  \
              -o  "/content/content/results/pseudo-nms" \
              -t 0.2

PYTHONPATH=. python vietocr/postprocess/merge_phone.py  \
            -i       "/content/content/results/pseudo-nms" \
            -o       "/content/content/results/pseudo-phone"        

PYTHONPATH=. python vietocr/postprocess/update_dict.py \
                     --txt-paths /content/content/results/pseudo-phone/* \
                     --threshold 0.1 \
                     --output-dir "/content/content/results/pseudo-dict"

PYTHONPATH=. python vietocr/postprocess/clean.py  \
            --labels       "/content/content/results/pseudo-dict" \
            --output       "/content/content/results/pseudo-final"      


zip -r /content/results/maskrcnn-fold.zip "/content/results/"