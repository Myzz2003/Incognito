PROJECT_PATH: str = "/Users/maoyingzhe/Desktop/Work/Project/Incognito"
TEMP_PATH: str = "/Users/maoyingzhe/Desktop/Work/Project/Incognito/temp"
CASCADE_CLASSIFIER_PATH: str = "/Users/maoyingzhe/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"

import cv2 as cv
import os
import sys
sys.path.append(PROJECT_PATH)
from utils import OptimizeImage

os.system("clear")
face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

imgs_dir = os.listdir(os.path.join(TEMP_PATH, "raw_faces"))

for imgd in imgs_dir:
    if imgd == ".DS_Store":
        continue
    img_path = os.path.join(TEMP_PATH, "raw_faces", imgd)
    img = cv.imread(img_path)
    if img is None:
        continue
    img_cnt = 0
    faces = face_cascade.detectMultiScale(img, 1.1, 3)
    for (x,y,w,h) in faces:
        face_area = img[y:y+h, x:x+w]
        img_name = f"added_{imgd.split('.')[0]}_{img_cnt}.jpg"
        img_cnt += 1
        cv.imwrite(os.path.join(TEMP_PATH, "faces", img_name), face_area)
        print(f"Face detected in {imgd}, saved at {os.path.join(TEMP_PATH, 'faces', img_name)}")

    

print(f"Face detection finished.")



