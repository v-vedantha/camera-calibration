from torch.utils.data import Dataset
import cv2
import os
import pickle
from find_eye_center import histogram, detect
import torch
from coordinate_mapping import Point, Iris, Input, MarkerPose, ChessBoard, Ellipse

class CalibrationDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if (index >= len(self.files)):
            index = 0
        folder_files = self.files[index]

        eye_file = os.path.join('calibration_images', folder_files, 'calibration_image_0.jpg')

        checkerboard_file = os.path.join('calibration_images', folder_files, 'calibration_image_2.jpg')

        mouse_file = os.path.join('mouse_positions', folder_files, 'positions.pkl')

        # Read mouse positions from pickle
        try:
            with open(mouse_file, 'rb') as f:
                mouse_position = pickle.load(f)
        except:
            return self.__getitem__(index + 1)

        # Read the two files
        eye_image = cv2.imread(eye_file)
        checkerboard_image = cv2.imread(checkerboard_file)

        # Compute their coordinates
        frame = cv2.flip(checkerboard_image, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nline = 6
        ncol = 9
        ret, corners = cv2.findChessboardCorners(gray, (nline, ncol), None)
        if not ret:
            return self.__getitem__(index + 1)

        params = {
            "threshold" : 63,
            "circularity" : 1.3,
            "extend" : 0.85,
            'area_min' : 10000,
            'area_max' : 25000,
            'left_boundary' : 100,
            'right_boundary' : 1000,
            'top_boundary' : 100,
            'bottom_boundary' : 1000,
            'previous_ellipse' : None
        }
        
        try:
            new_hist = histogram(eye_image[80:700, 500:1300])

            params['threshold'] = new_hist
        except Exception as e:
            print(e)
            return self.__getitem__(index + 1)
        detection = detect(eye_image, params)
        if detection == None:
            print("Detection missed")
            return self.__getitem__(index + 1)

        return torch.cat((ChessBoard(corners).convert_to_torch(), Ellipse(detection).convert_to_torch()), 0), Point(mouse_position[0], mouse_position[1]).convert_to_torch()

        return eye_image, checkerboard_image, mouse_position

class SimpleDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
