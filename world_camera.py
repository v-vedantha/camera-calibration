from types import FrameType
import cv2
import numpy as np
from coordinate_mapping import Point
import sys
sys.path.append("GazeTracking")
from gaze_tracking import GazeTracking
det = cv2.QRCodeDetector()
gaze = GazeTracking()
def detect_qr_codes(image):
    rv, pts = det.detectMulti(np.hstack([image])) 

    breakpoint()
    if pts.shape != (4, 4, 2):
        raise Exception("All 4 QR codes must be visible")

    centers = [tuple(i) for i in pts.sum(axis = 1) / 4]
    
    centers.sort()

    if centers[0][1] < centers[1][1]:
        top_left = Point(centers[0][0], centers[0][1])
        top_right = Point(centers[1][0], centers[1][1])
    else:
        top_left = Point(centers[1][0], centers[1][1])
        top_right = Point(centers[0][0], centers[0][1])
    if centers[2][1] < centers[3][1]:
        bottom_left = Point(centers[2][0], centers[2][1])
        bottom_right = Point(centers[3][0], centers[3][1])
    else:
        bottom_left = Point(centers[3][0], centers[3][1])
        bottom_right = Point(centers[2][0], centers[2][1])

    return top_left, top_right, bottom_left, bottom_right
def take_photo():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    return frame

def compute_cam():
    return detect_qr_codes(cv2.flip(take_photo(), 1))

def compute_world_cam(frame):
    gaze.refresh(frame)
    left_coords = gaze.pupil_left_coords()
    right_coords = gaze.pupil_right_coords()

    if left_coords is None or right_coords is None:
        return None
    return Point(left_coords[0], left_coords[1]), Point(right_coords[0], right_coords[1])

def world_cam_process(buffer, index_start, cam_index):
    camera = cv2.VideoCapture(cam_index)
    while True:
        ret, frame = camera.read()
        if frame is None:
            continue
        output = compute_world_cam(frame)
        if output is not None:
            frame = cv2.circle(frame, (int(output[0].x), int(output[0].y)), 3, (0, 0, 255), -1)
            print(output)
            cv2.imshow("face", frame)
            cv2.waitKey(1)
            output[0].move_to_shared_mem(buffer, index_start)
            output[1].move_to_shared_mem(buffer, index_start+1)
        


if __name__ == '__main__':
    print(compute_cam())
