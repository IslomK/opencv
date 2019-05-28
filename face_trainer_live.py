import os
from PIL import Image
import numpy
import cv2
import requests

url = 'http://192.168.3.2:8080/shot.jpg'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

font = cv2.FONT_HERSHEY_SIMPLEX
stroke = 2
x,y = [100,100]
color = ()
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
img_num = 0
folder_numbers = 0

for _, folders, files in (os.walk(image_dir)):
    folder_numbers += len(folders)
new_folder = os.path.join(image_dir, '{}'.format(folder_numbers + 1))
os.mkdir(new_folder)

while (True):
    text = "Face is not detected"

    # img_resp = requests.get(url)
    # img_array = numpy.array(bytearray(img_resp.content), dtype=numpy.uint8)
    # img = cv2.imdecode(img_array, -1)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        color = (0,0,0)
        img_item = "my-image.png"
        cv2.imwrite(img_item, gray)
        
        end_coord_x = x + w
        end_coord_y = y + h
    
        rectangle = cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)
        if len(faces)>0:
            img_num += 1
            color = (255,255,255)
            text = "Face detected"
            cv2.putText(frame, "Please don't move", (150,150), font, 1,color, stroke, cv2.LINE_AA)        
            if img_num % 2 == 0:
                cv2.imwrite(os.path.join(image_dir, str(folder_numbers + 1), 'image{}.png'.format(img_num)), roi_gray)
    
    cv2.putText(frame, text, (x,y), font, 1,color, stroke, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        os.rename(new_folder, os.path.join(image_dir, input("Enter your name please: ")))
        break

import face_trainer