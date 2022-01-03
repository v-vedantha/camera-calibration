import numpy as np
import torch
import cv2

from coordinate_mapping import ChessBoard
import sys
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def chessboard_process(buffer, index, cam_index):
    camera = cv2.VideoCapture(cam_index)
    nline = 6
    ncol = 9
    while True:
        frame = camera.read()[1]
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nline, ncol), None)
        if not ret:
            cv2.imshow('MarkerPose', frame)
            cv2.waitKey(1)
            continue
        for corner in corners:
            frame = cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 3, (0, 0, 255), -1)
        cv2.imshow('MarkerPose', frame)
        cv2.waitKey(1)
        # Show image
        # Write to shared memory
        ChessBoard(corners).move_to_shared_mem(buffer, index)




