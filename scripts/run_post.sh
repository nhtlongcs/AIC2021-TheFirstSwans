PREDICTED_FOLDER=$1
OUTPUT_FOLDER=$2
rm -rf ./tmp/
mkdir -p ./tmp/
mkdir -p OUTPUT_FOLDER

PYTHONPATH=paddleocr/transformer  python postprocess/split_phrase_to_word.py \
                            $PREDICTED_FOLDER \
                            "./tmp/phrase1"

PYTHONPATH=paddleocr/transformer  python postprocess/split_phrase.py \
                            "./tmp/phrase1" \
                            "./tmp/phrase2"

PYTHONPATH=paddleocr/transformer  python postprocess/split_box_by_token.py  \
                            --text   "./tmp/phrase2" \
                            --output "./tmp/splitted"

PYTHONPATH=paddleocr/transformer  python postprocess/fuse.py  \
                            -i  "./tmp/splitted"  \
                            -o  "./tmp/fuse" \
                            -t 0.2

PYTHONPATH=paddleocr/transformer  python postprocess/merge_phone.py  \
                            -i  "./tmp/fuse" \
                            -o  "./tmp/phone"        

PYTHONPATH=paddleocr/transformer  python postprocess/update_dict.py \
                            --txt-paths ./tmp/phone/* \
                            --threshold 0.1 \
                            --output-dir "./tmp/dict"

PYTHONPATH=paddleocr/transformer  python postprocess/clean.py  \
                            --labels    "./tmp/dict" \
                            --output    $OUTPUT_FOLDER     
