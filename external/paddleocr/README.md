# PaddleOCR

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">

## Installation

```
cd external/paddleocr
pip install -r requirements.txt
pip install paddleocr
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

## Data Format

The provided annotation file format is as follow, seperated by "\t":

```
Image file name                 Image annotation information encoded by json.dumps"
vin-scene-text/img_1.jpg    [{"transcription": "VIỆT", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

The image annotation after json.dumps() encoding is a list containing multiple dictionaries.

The points in the dictionary represent the coordinates (x, y) of the four points of the text box, arranged clockwise from the point at the upper left corner.transcription represents the text of the current text box. When its content is "###" it means that the text box is invalid and will be skipped during training.


## Train

- Before training, modify config file 

- Notes: Sometimes during the training, the notebook will crash, due to RAM overload, if you face this problem, try change the image size in the config for both training and evaluation phase. (there are some images in the Vin dataset that have large sizes)


```
python tools/train.py \
         -c <path to config> 
         -o Global.checkpoints= <resume checkpoint>  \
         Global.save_model_dir = <saved path> \
```

## Inference

```
python tools/infer_det.py  -c <path to config> \
                            -o Global.checkpoints=<trained checkpoint> \
                            Eval.loader.num_workers=0 \
                            PostProcess.box_thresh=<iou threshold> \
                            PostProcess.unclip_ratio=1.5 \
                            PostProcess.thresh=<conf threshold> \
                            Global.infer_img=<path to image folder> \
                            Global.save_res_path=<saved result path> \
```



## Results

- Evaluation on pseudo B1

| Model         | Precision | Recall | Hmean |Checkpoint
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| ppocrv2-db | 0.79378	| 0.8222 | 0.807 | [link](https://drive.google.com/file/d/1DvA0J-Hoj99a43kxtR6rwIiwAQjq2dOx/view?usp=sharing)


## References
- Modified from: https://github.com/PaddlePaddle/PaddleOCR


