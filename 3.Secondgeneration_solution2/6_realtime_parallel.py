# ========================== Import block ===========================
import cv2
import csv
import os
import time
import mediapipe as mp
import sqlite3
import numpy as np
from datetime import datetime
from music21 import pitch, note, stream, midi
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax

# packages for multi-threading
import threading
import queue
from collections import deque
import sys
# ========================== Import block ===========================


# ========================== Control Params =========================
n_note = 20
use_model = True
# ========================== Control Params =========================


pTime = 0 #处理一张图像前的时间
cTime = 0 #一张图处理完的时间

# 存放坐标信息
lmList = []
captured_coordinates = []

model = Sequential()
#Adding layers
model.add(LSTM(512, input_shape=(40, 1), return_sequences=True)) #添加第一个长短时记忆（LSTM）层，其中有512个神经元。LSTM层通常用于处理序列数据。
#X shape is(L_datapoints, length, 1)，
#X.shape[1]and X.shape[2] means the second and the third feature of X shape，which are ‘length’and‘1’
#X的形状（shape）是指X这个数组在各个维度上的大小。在NumPy中，一个数组的形状可以通过其shape属性来访问，它是一个元组，其中包含了数组在每个维度上的大小。
#例如，如果X是一个三维数组，其中第一个维度有L_datapoints个元素，第二个维度有length个元素，第三个维度有1个元素，那么X的形状就是(L_datapoints, length, 1)。
model.add(Dropout(0.1))
#Dropout Layer: This layer is a special type of layer used to "drop" a random fraction of the neurons' outputs during training. 
#By doing so, the network is forced to learn more robust and generalized representations.
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.1))
model.add(Dense(229, activation='softmax'))
#Compiling the model for training  
opt = Adamax(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.load_weights("model_X+notes.h5")

import pickle
with open('mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

with open('reverse_mapping.pkl', 'rb') as f:
    reverse_mapping = pickle.load(f)

with open('L_corpus.pkl', 'rb') as f:
    L_corpus = pickle.load(f)

with open('L_symb.pkl', 'rb') as f:
    L_symb = pickle.load(f)
    
with open('Corpus.pkl', 'rb') as f:
    Corpus = pickle.load(f)

#Splitting the Corpus in equal length of strings and output target
length = 40
features = [] # 这两行创建了两个空列表，分别用来存储训练数据的输入特征（即音符序列）和目标（即接下来的音符）。
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])
    
    
L_datapoints = len(targets)
print("Total number of sequences in the Corpus:", L_datapoints)

# reshape X and normalize
X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)
# one hot encode the output variable
y = tensorflow.keras.utils.to_categorical(targets) 
X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)
seed = X_seed[0]

def generate_notes(model, seed_sequence, num_notes_to_generate):
    generated_notes = []
    current_sequence = np.array(seed_sequence)
    
    for _ in range(num_notes_to_generate):
        prediction = model.predict(current_sequence.reshape(1, length, 1))
        predicted_note_index = np.argmax(prediction)
        generated_notes.append(reverse_mapping[predicted_note_index])
        
        # 更新当前序列以进行下一次预测
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_note_index

    return generated_notes

def create_note_by_midi(m, dur=1):
    n = note.Note(midi=m)  # This is Middle C
    n.duration.type = 'half'  # Make it a half note (default is quarter)
    n.duration.quarterLength = dur
    return n

def create_note(n, dur=1):
    n = note.Note(n)  # This is Middle C
    n.duration.type = 'half'  # Make it a half note (default is quarter)
    n.duration.quarterLength = dur
    return n

def play_note(n):
    s = stream.Stream()
    s.append(n)
    sp = midi.realtime.StreamPlayer(s)
    sp.play()
    return

#mapping X coordinates and decide the duration
def map_x_to_duration(x):
    # 定义映射的范围
    durations = [4, 2, 1, 0.5, 0.25]
    segment = 1.0 / len(durations)

    # 根据x值确定其所属的段，并返回对应的duration
    for i, duration in enumerate(durations):
        if x < (i + 1) * segment:
            return duration

def adjust_to_c_major(note_name):
    c_major_scale = ["C", "D", "E", "F", "G", "A", "B"]
    
    # Check if note_name is valid
    try:
        given_pitch = pitch.Pitch(note_name)
    except:
        # Return a default note
        return "C4"
    
    if given_pitch.name in c_major_scale:
        return note_name
    
    # 使用半音上升或下降来寻找最近的C大调音符
    up = given_pitch.transpose(1)
    down = given_pitch.transpose(-1)
    
    if up.name in c_major_scale:
        return up.nameWithOctave
    elif down.name in c_major_scale:
        return down.nameWithOctave
    else:
        return note_name
    

notes = []
if use_model:
    notes = generate_notes(model, seed, n_note);
notes



# 创建保存数据的目录
output_dir = "output_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def capture_and_render(note_info, done_info):
    print("in caputure and render")
   
    played_note = []
    note_idx = 0
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, 
                        max_num_hands=2,
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)  
    mpDraw = mp.solutions.drawing_utils
    normalized_coords = []
    pixel_coords = []
    lmList = []  
    pTime = 0
    while True:
        
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                for index, lm in enumerate(handlms.landmark):
                    if index == 8:  
                        print("In inner loop")
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        midi_pitch = int(lm.y * 60 + 30)
                        # dur = lm.x * 4 + 0.1
                        dur = map_x_to_duration(lm.x)
                        print("after process")


                        done = None
                        if (not done_info.empty()):
                            done = done_info.get()
                        else:
                            print("done_info queue empty")
                            done = False
                        print("Done is:", done)
                        if (done):
                            n = None
                            if (not use_model):
                                n = create_note_by_midi(midi_pitch, dur)
                            else:
                                n = create_note(notes[note_idx], dur)
                                note_idx += 1
                            n_name = n.nameWithOctave
                            adjusted_n_name = adjust_to_c_major(n_name)
                            final_note = create_note(adjusted_n_name, dur)
                            # ==========================
                            print("sending note:", final_note)
                            note_info.put(final_note)
                            # ==========================
                        cv2.circle(img, (cx, cy), 12, (0, 220, 23), cv2.FILLED)
        # FPS显示和其他后处理
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == 27:
            # 释放摄像头资源
            cap.release()
            # 关闭所有 OpenCV 窗口
            cv2.destroyAllWindows()
            break


def play_music(note_info, done_info):
    while True:
        if (not note_info.empty()):
            note = note_info.get()
            print("Got it!", note)
            play_note(note)
            # note_info.clear()
            done_info.put(True)
            print("done info written:", done_info)
        else:
            # done_info.put(False)
            pass


if ( __name__ == "__main__"):

    # Queues for multi-threading info passing
    # note_info = queue.Queue()
    # done_info = queue.Queue()

    note_info = queue.Queue()
    done_info = queue.Queue()
    done_info.put(True)

    t1 = threading.Thread(target=capture_and_render, args=(note_info, done_info))
    t2 = threading.Thread(target=play_music, args=(note_info, done_info))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    sys.exit()

