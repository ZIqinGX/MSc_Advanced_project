Below are code for **hand position capture**. The primary function of this section of code is to detect the position of a human hand and obtain its coordinates. Once the hand coordinates are acquired, the attributes of the music can be adjusted based on these coordinates.Further process these coordinates, such as mapping them to the tempo or other attributes of the music.

## 1. First step: Install CV2 and mediapipe
Run the following command in the terminal
```
pip install opencv-python
pip install opencv-contrib-python  # Including extra module
pip install mediapipe
```
MediaPipe is an open-source framework for building pipelines to perform computer vision inference over arbitrary sensory data such as video or audio.
Read more at: https://viso.ai/computer-vision/mediapipe/
