import cv2
import numpy as np
import os
import requests
import pickle

labels =  {}
with open('lables.pickle', "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

url = 'http://192.168.3.2:8080/shot.jpg'


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')


while(True):

    img_resp = requests.get(url)
    img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=40 and conf<=85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            


        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (0, 0, 0)
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h

        cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)

    cv2.imshow('frame', img)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
