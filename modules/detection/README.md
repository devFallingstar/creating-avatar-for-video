# face detection and identification


------

## 1. Intro

- This repo's origin code from (https://github.com/TreB1eN/InsightFace_Pytorch)

- python 3.6

- nvidia driver : 410.78

- cuda : 10.0

- pytorch : 1.4.0

------

## 2. Pretrained Models

Pre-trained model is here [Google drive](https://drive.google.com/drive/folders/1S8w9ADr_86iEkWklhE8jfvctT5vzmlnr?usp=sharing)

Mobilefacenet and Ir_se

### 2.1 set configuration

set the model configuration in config.py line 19

```
conf.use_mobilfacenet = True
```
if True, you will use mobliefacenet or else

## 3. How to use


### 3.1 Data Preparation

#### 3.1.1 Prepare Facebank (For testing over camera or video)

Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:

```
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```
I prepare facebank for news and parasite videos. please name it as facebank

The videos are not provided


#### 3.1.2 download the pretrained model to work_space/save

model file name as 'final_model' for GPU or 'final_model_cpu' for CPU

provided pre trained models are GPU version

### 3.2 detect over video:

```
​```
python infer_on_video.py -f [video file name] -s [save file name]
​```
```

the video file should be inside the data/face_bank folder

- Video Detection Demo [@Youtube](https://www.youtube.com/watch?v=6r9RCRmxtHE)

### 4. Future work

change face detection network
