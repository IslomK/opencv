import cv2
import numpy as np
import os
import pickle
from imutils.video import VideoStream

from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
from imutils.video import FPS

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

labels =  {}
with open('lables.pickle', "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

#cap = cv2.VideoCapture(0)

while(True):
    #ret, frame = cap.read()
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=500)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=40  and conf<=60:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            conf = '{0}%'.format(round(100 - confidence))
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            cv2.putText(frame, conf, (x,y), font, 1, color, stroke, cv2.LINE_AA)
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

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

