CASCADE_CLASSIFIER_PATH: str = "/Users/maoyingzhe/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"

import cv2 as cv
import os
import time
import torch
from utils import OptimizeImage, GetImageDescriptor, KNNMatch, MergeDescriptors, SelectROI
from nn import NetworkProcessor as npr

os.system("clear")
video = cv.VideoCapture(1)
face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)


net = npr("./sorted_faces", (64, 64), "./model.pt")
# net.trainNetwork()

face_area = None
sift = cv.SIFT_create()
bf = cv.BFMatcher()

print(f"Utilities Loaded.")

des_dict = GetImageDescriptor(sift, "./faces")
des_dict = MergeDescriptors(des_dict, threshold=10)

print(f"Descriptors Loaded.")

# first frame process to get basic parameters
ret, frame = video.read()
hw_ratio = frame.shape[0] / frame.shape[1]
height = 300
width = int(height / hw_ratio)
size_frame =  width * height
min_interest_size = size_frame / 8


while True:
    start_time = time.time()
    ret, frame = video.read()
    if not ret:
        raise Exception("Can't receive frame (stream end?). Exiting ...")

    frame = cv.resize(frame, (width, height))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    grab_face_flag = False
    for (x,y,w,h) in faces:
        if w*h > min_interest_size:
            grab_face_flag = True
            face_area = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            break
        else:
            continue

    if grab_face_flag:
        res = net.getLabelFromNet(face_area)
        res_label = res["label"]
        res_confidence = res["confidence"]
        if res_confidence > 0.5:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, f"{res_label}: {res_confidence:.2f}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    end_time = time.time()
    cv.putText(frame, f"FPS: {1 / (end_time - start_time):.0f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv.imshow("frame", frame)
    
    # if any key pressed, break
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

print(f"FPS: {1 / (end_time - start_time):.2f}")


video.release()


    