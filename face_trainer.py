import os
from PIL import Image
import numpy
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in (os.walk(image_dir)):
    for file_img in files:
        if file_img.endswith("jpg") or file_img.endswith("png"):
            path = os.path.join(root, file_img)
            label = os.path.basename(root).replace(" ", "-").lower()
            # x_train.append(path)
            # y_labels.append(label)

            pil_image = Image.open(path).convert("L") #greyscale
            size = (225, 225)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = numpy.array(pil_image, "uint8")

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=3)
            print(len(faces))

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(label_ids)
print(y_labels)


with open('lables.pickle', "wb") as f:
    pickle.dump(label_ids, f) 

recognizer.train(x_train, numpy.array(y_labels))
recognizer.save("trainer.yml")
