import numpy as np
import torch
import cv2

import sys
import os
sys.path.append('MarkerPose/Python/modules')
sys.path.append('MarkerPose/Python')
import models
import utils
from coordinate_mapping import MarkerPose


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def marker_pose_process(buffer, index):
    root = os.path.relpath('MarkerPose/dataset')

    superpoint = models.SuperPointNet(3)
    superpoint.load_state_dict(torch.load(os.path.join(root,'py_superpoint.pt'), map_location=device))

    # Create EllipSegNet model
    ellipsegnet = models.EllipSegNet(16, 1)
    ellipsegnet.load_state_dict(torch.load(os.path.join(root,'py_ellipsegnet.pt'), map_location=device))

    # Create MarkerPose model
    markerpose = models.MarkerPose(superpoint, ellipsegnet, (320,240), 120)
    markerpose.to(device)
    markerpose.eval()
    camera = cv2.VideoCapture(1)
    while True:
        frame = camera.read()[1]
        if frame is None:
            continue
        # Hacky use of markerpose, but it works for now
        centers = markerpose(frame, frame)

        if centers == (None, None):
            continue

        # Draw centers
        for center in centers:
            frame = cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

        # Show image
        cv2.imshow('MarkerPose', frame)
        cv2.waitKey(1)
        # Write to shared memory
        MarkerPose(centers).move_to_shared_mem(buffer, index)




