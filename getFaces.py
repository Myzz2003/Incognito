XML_PATH = "/Users/maoyingzhe/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"

import cv2 as cv
import os
import numpy as np

video = cv.VideoCapture(1)
face_cascade = cv.CascadeClassifier(XML_PATH)

ret, frame = video.read()
hw_ratio = frame.shape[0] / frame.shape[1]
height = 300
width = int(height / hw_ratio)
size_frame =  width * height
min_interest_size = size_frame / 8

while True:
    ret, frame = video.read()
    if not ret:
        RuntimeError("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.resize(frame, (width, height))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    grab_face_flag = False
    for (x,y,w,h) in faces:
        if w*h > min_interest_size:
            grab_face_flag = True
            face_area = frame[y:y+h, x:x+w]
            break
        else:
            continue

    if grab_face_flag:
        # get directory file list
        file_list = os.listdir("./faces")
        # name the file with format "face_{n}.jpg"
        file_list = [file for file in file_list if file.startswith("face_")]
        file_list = sorted(file_list, key=lambda x: int(x.split("_")[1].split(".")[0]))
        # get the last file name
        if len(file_list) == 0:
            last_file_name = "face_0.jpg"
        else:
            last_file_name = file_list[-1]
        # get the last file number
        last_file_num = int(last_file_name.split("_")[1].split(".")[0])
        # save the face
        cv.imwrite(f"./faces/face_{last_file_num+1}.jpg", face_area)
        print(f"face_{last_file_num+1}.jpg saved")
        break

cv.imshow("face", face_area)
cv.waitKey(0)

video.release()