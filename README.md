# MSc_Advanced_project
![图片名称](https://raw.githubusercontent.com/ZIqinGX/MSc_Advanced_project/main/pictures/sheet1.png) <br/>

Hi! This is repository for my Advanced project. It is the final course of my graduate career. This repository includes Readme about this project, code of this project and video link. 
It is a project about machine learning and human computer interaction.I want to explore the collaboration between humans and artificial intelligence in improvisation and real-time performances.<br/>

![pic](https://raw.githubusercontent.com/ZIqinGX/MSc_Advanced_project/main/pictures/finalwork2.png)<br/>

**Video Link：** https://youtu.be/WQ1uSLL0rSo<br/>

**What is my project about?**
The motivation of creating this porject is I want to have a tool for people without music background like me to enjoy the pleasure of creating music. In this project, I designed a AI and human co-create realtime-system for music generation based on reflecting existed art work. This system provides a tool to manipulate duration of notes. Notes is predicted by LSTM. Hand coordinates is captured by MediaPipe framework. Hand coordinates will be mapped to corresponding duraion according to its value.

**How to read this repository**
It goes along the time sequence and the design iteration of my project.


**Design process and iteration**
The figure below shows plan of **First generation**.<br/>

![图片名称](https://raw.githubusercontent.com/ZIqinGX/MSc_Advanced_project/main/pictures/First_generation.jpg) <br/>
Code for first generation<br/>
https://github.com/ZIqinGX/MSc_Advanced_project/tree/main/First_Generation


The figure below shows plan of the first solution for **Second generation**.<br/>

![图片名称](https://raw.githubusercontent.com/ZIqinGX/MSc_Advanced_project/main/pictures/Second_generation_1.jpg) <br/>



For more detailed development, please check my **Weekly blog**: https://github.com/ZIqinGX/MSc_Advanced_project/blob/main/Weekly_blog/Weekly_document.md <br/>

For **environment requirements**, please check: https://github.com/ZIqinGX/MSc_Advanced_project/blob/main/requirements.txt <br/>

## **Use tip:**
To use this system, you can check **4.Final_outcome** https://github.com/ZIqinGX/MSc_Advanced_project/tree/main/4.Final_outcome<br/>
First use **1LSTMmusic.ipynb**<br/>
Then run code **2_Build_model.ipynb**<br/>
Finally, run **3_realtime_parallel.py**<br/>






By the way,<br/>
This notebook helped me a lot  **Music Generation: LSTM** <br/>
(from https://www.kaggle.com/code/karnikakapoor/music-generation-lstm/notebook) <br/> I put the notebook containing the code which I repeat the process of this notebook in LSTM_learning_from_Kaggle_Comment.ipynb https://github.com/ZIqinGX/MSc_Advanced_project/blob/main/Reproduce_the_code_from_Kaggle/LSTM_learning_from_Kaggle_Comment.ipynb<br/> (You can find it in the file called *Reproduce the code from Kaggle*)<br/>
**It is not my outcome but my learning process and my notes.** <br/>It is more like trying to figure out how to generate music by machine learning model from the ground. 
The reasons for putting it here are mainly two aspects. First is I want to document my learning process and explanations of code to facilitate future review and reference. Second is for people who see this repository and also have interest in machine learning music generating, but don't know where to start（just like me）. I want to keep a set of study notes for learning, exchanging ideas, and sharing. <br/>
My knowledge of how to read music sheet comes from：https://www.flowkey.com/zh/piano-guide/read-sheet-music
