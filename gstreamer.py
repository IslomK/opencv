import numpy as np
import cv2
from multiprocessing import Process


def receive():
    cap_receive = cv2.VideoCapture('ffplay -analyzeduration 1 -fflags -nobuffer -i udp://10.10.1.128:5000')

    if not cap_receive.isOpened():
        print('VideoCapture not opened')
        exit(0)
    
    while True:
        ret,frame = cap_receive.read()

        if not ret:
            print('empty frame')
            break

        cv2.imshow('receive', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap_receive.release()

if __name__ == '__main__':
    r = Process(target=receive)
    r.start()
    r.join()

    cv2.destroyAllWindows()