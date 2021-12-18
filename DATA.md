# **DATASET**


## **Text Detection**

Format: Each image has an annotation text file. Each line in that file correspond to the polygon and text

```
x1,y1,x2,y2,x3,y3,x4,y4,text
```

### Vin Scene Text
- This has been cleaned and fixed from original dataset
- [Download link](https://drive.google.com/file/d/1dNKWqN7vYtXWC1-OYwWwWW20zkKuHaDc/view?usp=sharing)
- Some samples:

<img height="300" alt="screen" src="./assets/data/det/vin/im1629.jpg"> 
<img height="300" alt="screen" src="./assets/data/det/vin/im1798.jpg"> 
<img height="300" alt="screen" src="./assets/data/det/vin/im1803.jpg"> 
<img height="300" alt="screen" src="./assets/data/det/vin/im1212.jpg"> 

### Vietnamese Synth Text:
- This is synthesized from original SynthText and ICDAR15 background 
- [Download link](https://drive.google.com/file/d/1it9fOUnym9r0N4luSsSuD7ZhrFMHT0oc/view?usp=sharing)
- Some samples:

<img height="250" alt="screen" src="./assets/data/det/synth/image-000000001.png"> 
<img height="250" alt="screen" src="./assets/data/det/synth/image-000003153.png"> 
<img height="250" alt="screen" src="./assets/data/det/synth/image-000007959.png"> 
<img height="250" alt="screen" src="./assets/data/det/synth/image-000009950.png">
<img height="250" alt="screen" src="./assets/data/det/synth/image-000016530.png">
<img height="250" alt="screen" src="./assets/data/det/synth/image-000016534.png">

## **Text Recognition**

- Image and text are cropped from above datasets, using warp perspective to extract the area in polygons

### Vin Scene Text:
- [Download link](https://drive.google.com/file/d/1pmuT0z0bUWcVdiOVKkmZSUkCMpaej2kn/view?usp=sharing)

<img height="300" alt="screen" src="./assets/data/det/vin/im1798.jpg"> 
<br>
<img height="20" alt="screen" src="./assets/data/rec/im1798/1.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/2.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/3.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/4.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/5.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/6.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/7.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/8.png"> 
<img height="20" alt="screen" src="./assets/data/rec/im1798/9.png"> 



### Vietnamese Synth Text:
- [Download link](https://drive.google.com/file/d/1jUf8IdFPWnmnTxq12KtArn2SqiKIV6hH/view?usp=sharing)

<img height="300" alt="screen" src="./assets/data/det/synth/image-000016530.png"> 
<br>
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/1.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/2.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/3.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/4.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/5.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/6.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/7.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/8.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/9.png"> 
<br>
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/10.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/11.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/13.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/14.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/15.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/16.png"> 
<img height="20" alt="screen" src="./assets/data/rec/image-000016530/12.png"> 