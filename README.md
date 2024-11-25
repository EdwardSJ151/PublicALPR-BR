# YoloALPR
The objective of this repo is to create effective methods of ALPR for brazilian license plates, using YOLO as the backbone of the project.

### How to Use
1. Clone the repo:
```bash
git clone https://github.com/EdwardSJ151/PublicALPR-BR.git
```

2. Add you images/videos in the `imgAndVideos` directory

3. Run `main.py` using image inference, video inference or webcam inference:
```bash
python main.py --mode image --input_path ./imgAndVideos/img1.jpg --network_size 992,736
```
```bash
python main.py --mode video --input_path ./imgAndVideos/video1.mp4
```
```bash
python script.py --mode webcam --network_size 992,736
```

**Note:** The current model is only available in the network_size of `992, 736`. If an issue is ever opened for another dimension size, I will train a model for that size with an adapted version of the dataset to best fit it and add it here.

**Disclaimer:** If the `ESC` key isn't used to exit out of the cv2 GUI, your machine will not close it correctly, leading to issues.

### What is YOLO and YOLOv4-tiny, and why use it?
YOLO (You Only Look Once) is an Convolutional Neural Network object detection algorithm which, unlike other object detection algorithms that have a pipeline where the objects in the image need to have their location detected, and only then they can be classified, YOLO embraces the "You Only Look Once" title and does the detection in a single step, effectively reducing processing time for inference drastically compared to other models but maintaining competitive results, which is why it is being used. Since with most ALPR application, quick inference is needed, this is a great model to use, and will be what most of the project will use. 

<p align="center">
  <img width=500, src="https://github.com/EdwardSJ151/PublicALPR-BR/assets/88259575/8a00c1e6-e937-4e15-b070-e930d010aa09" alt="Basic_Yolo_Architecture_Explanation">
</p>

Yolo has had many versions release over the years, and for this project, specifically Yolov4-tiny will be used. Yolov4-tiny is based on CSPDarknet53, which is inspired by the base Yolov4 model. CSPDarknet53 is an algorithm that tries to improve upon YOLOv4 by cutting down on its memory footprint and raising efficiency by dividing the input data into smaller sections that are easier to process. It then uses cross-stage connections to combine these sections back into a larger representation of the input data. This approach improves the accuracy of the network while reducing the amount of system memory (RAM) required to store the intermediate representations of the input. This is important to run the model on low grade hardware, but still maintain a high frame rate.

Maintating such high efficiency is important specifically for my use case, where I want to run the model on a single board computer, or SBC for short, (Raspberry Pi, Orange Pi, NVIDIA Jetson Nano, etc...) to do inference locally.

### Dataset
For this first update, I currently only have 122 images. About 60% of the images contains images with plates, while the rest contains negative samples. With data augmentation techiques, I have a total of around 400 images. I am creating the entire dataset by scratch, using some images from the internet and taking pictures of the license plates from my phone (S23). 

<p align="center">
  <img width=550, src="https://github.com/EdwardSJ151/PublicALPR-BR/assets/88259575/328dc2d1-846a-469b-8049-7880173bf0a5" alt="Dataset_Example">
</p>

I still plan on adding upon the dataset in future updates. Since I am creating the dataset with mainly images I am taking myself, I will not release the dataset due to privacy reasons. Currently there is a class disbalance due to the small number of images, and my current objective is to have a more balanced dataset with at least 1000 total images (without counting data augmentation generated images).

### ALPR Pipelines
#### Pipeline 1

<p align="center">
  <img width=450, src="https://github.com/EdwardSJ151/PublicALPR-BR/assets/88259575/4e27612c-0d08-482c-bf3f-095f1875a643" alt="Pipeline 1">
</p>

With this pipeline, the only model to be used will be YOLO. This is a unexplored method, with the more common approach being pipeline 2s approach. Here, along with treating the license plates as being objects to be detected, the characters are also treated as objects to be detected by YOLO. A couple of frames with predictions are collected, and then all of the prediction frames "vote" and the prediction with the most "votes" is the final prediction. This technique has it's advantages, specifically faster inference times due to only having one step for detection, but also being able to identify rotated characters with ease.

<p align="center">
  <img width=550, src="https://github.com/EdwardSJ151/PublicALPR-BR/assets/88259575/a9393e3b-6d60-473c-ad48-6075098e2ee2" alt="Pipeline 1">
</p>

Although, there are disadvantages. YOLO can have dificulty identifying the characters if the frame shows the character at an angle that isn't well represented in the dataset, and with so many different object classes, your dataset needs to reflect all possible angles that can be seen. With this knowledge, this pipeline is useful if you have a fixed camera angle. I want to try and reduce the disadvantages of this pipeline as much as possible by trying to create a good dataset. My application will have a fixed camera angle, and I need extreamely fast inference, which makes this pipeline ideal for such necessities.

#### Pipeline 2

<p align="center">
  <img width=650, src="https://github.com/EdwardSJ151/PublicALPR-BR/assets/88259575/7a289987-7e8e-403a-8561-a90babe4cdde" alt="Pipeline 2">
</p>

This second pipeline represents a more common approach to ALPR. Using YOLO, the license plate is detected, and then everything outside of the prediction bounding box is cut out. With just the bouding box, character segmentation techiques are excecuted to seperate the characters in order. This is done so we can have the individual characters and help with the next step of the pipeline, which is character recognition. After getting the final prediction of the license plate, this process is redone multiple times to get multiple frames with one prediction each, just like previously. Then, the vote happens and the final prediction is made.

### Progress

#### Done:
- Clearly defined the model and techniques to be used.
- Creation of two pipelines to experiment with (Only Mercosul plates).
- Creation of first test dataset to experiment with pipeline 1. (Around 120 images)
- Creation of small script for model inference via image, video or webcam.

#### In progress
- Expantion of the dataset to make a reliable model and also experiment pipeline 2. 500 images minimum, 1000 images is the goal for flexibility.
- Creation of a small script to run inference easily via CLI.
- Support the old plate format

#### To Do
- Research a way to maybe create a better, more efficient, third pipeline.
- Create a dockerfile for the project.
- After testing inference on the SBC, maybe experiment with Yolov4, Yolov43L, or other tiny versions of YOLO.
