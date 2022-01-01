import numpy as np
import torch
import cv2
import sys
sys.path.append('/Users/vedantha/Desktop/camera-calibration/irislandmarks_pytorch/')
from irislandmarks_pytorch.irislandmarks import IrisLandmarks
from coordinate_mapping import Iris
def iris_landmark_process(buffer, index_start):
    
    net = IrisLandmarks()
    net.load_weights("/Users/vedantha/Desktop/camera-calibration/irislandmarks_pytorch/irislandmarks.pth")
    camera = cv2.VideoCapture(1)

    while True:
        frame = camera.read()[1]
        if frame is None:
            continue
        frame = np.asarray(cv2.resize(cv2.flip(frame, 1), (64, 64)))
        eye, iris = net.predict_on_image(frame)
        frame = cv2.circle(frame, (int(iris.flatten()[0]), int(iris.flatten()[1])), 3, (0, 0, 255), -1)
        cv2.imshow("iris", frame)
        cv2.waitKey(1)
        Iris(eye, iris).move_to_shared_mem(buffer, index_start)

        

if __name__ == '__main__':
    iris_landmark_process(print)

