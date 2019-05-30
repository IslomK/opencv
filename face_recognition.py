import cv2
import numpy as np
import os
import pickle

labels =  {}
with open('lables.pickle', "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=40 and conf<=60:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = 'Unknown face detected'
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (0, 0, 0)
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h

        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)
    
    for x,y,w,h in eyes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
