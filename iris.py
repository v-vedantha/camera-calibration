import numpy as np
import torch
import cv2
import sys
sys.path.append('irislandmarks_pytorch/')
from irislandmarks_pytorch.irislandmarks import IrisLandmarks
from coordinate_mapping import Iris
def iris_landmark_process(buffer, index_start, cam_index, mapper):
    
    net = IrisLandmarks()
    net.load_weights("irislandmarks_pytorch/irislandmarks.pth")
    camera = cv2.VideoCapture(cam_index)

    while True:
        frame = camera.read()[1]
        if frame is None:
            continue
        frame = np.asarray(frame)
        frame = frame[200:, 800:1600]
        frame = np.asarray(cv2.resize(cv2.flip(frame, 1), (64, 64)))
        eye, iris = net.predict_on_image(frame)

        num_points = 0
        for p in iris[0,:,:2]:
            frame = cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
        cv2.imshow("iris", cv2.resize(frame, (320, 240)))
        cv2.waitKey(1)
        mapper(eye, iris).move_to_shared_mem(buffer, index_start)

        

if __name__ == '__main__':
    iris_landmark_process(print)

