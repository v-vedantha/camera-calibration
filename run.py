from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory, ShareableList
from coordinate_mapping import Point, Iris, Input

from world_camera import world_cam_process
from iris import iris_landmark_process
from marker_pose import marker_pose_process
from chessboard_marker import chessboard_process
import numpy as np
import sys
import pickle
import cv2
cv2.namedWindow("iris", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("MakerPose", cv2.WINDOW_AUTOSIZE)
#cv2.waitKey(1)
sys.path.append("/Users/vedantha/Desktop/camera-calibration/eyeloop/eyeloop/extractors")
import custom


def build_buffer():
    return ShareableList(['-'*10000, '-'*10000])


def main():
    buffer = build_buffer()
    #iris_landmark_process(buffer, 0)
    custom_extractor = custom.custom_Extractor(buffer=buffer)
    #w = Process(target=world_cam_process, args=(buffer, 0))
    #chessboard_process(buffer, 0)
    m = Process(target=chessboard_process, args=(buffer, 0, 1))
    p = Process(target=iris_landmark_process, args=(buffer, 1, 0))
    m.start()
    p.start()
    while True:
        pass
if __name__ == '__main__':
    main()
