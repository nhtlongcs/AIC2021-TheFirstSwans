# TransformerOCR


<p align="center">

| <img width="512" height="614" alt="screen" src="https://raw.githubusercontent.com/pbcquoc/vietocr/master/image/vietocr.jpg"> |
| :----------------------------------------------------------: |
|  Transformer Architecture | 

</p> 


## Installation

```
cd external/text-reg/vietocr
pip install -r requirement.txt
```

## Data Format

The provided annotation file format is as follow, seperated by "\t":

```
Image file name                 Text
vin-scene-text/img_1.jpg        VIỆT
```

## Train

- Modify config before train, specify dataset path
```
python train.py --config <config_path> \
                --checkpoint <checkpoint to resume>
```

## Inference
```
python infer.py 
```

- **Extra Parameters**:
    - ***--images***:       path to images folder
    - ***--labels***:       path to detection annotation folder
    - ***--output***:       path to output ocr annotation folder
    - ***--teencode***:     whether to use teen code
    - ***--weight***:       path to checkpoint
    - ***--config***:       path to config
    - ***--beam_search***:     whether to use beam search
    - ***--conf_threshold***:     ocr confidence threshold


## Results
| Backbone         | Config           | Precision full sequence | Checkpoint
| ------------- |:-------------:| ---:| ---:|
| VGG19-bn - Transformer | [vgg_transformer](https://drive.google.com/file/d/1-QqnSZbfxjUgQh3ir3STqx-RQ1Gu5Pr6/view?usp=sharing) | 0.85 | [link](https://drive.google.com/file/d/1-XsMJl0UNpAQlYb87K7USlWTe_4LM0DG/view?usp=sharing)


## References
- Modified from: https://github.com/pbcquoc/vietocr
