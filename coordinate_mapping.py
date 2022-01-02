import torch
import numpy as np
import pickle

#class Calibration():
#
#    def __init__(self, screen_w, screen_h, camera_w, camera_h):
#        self.screen_w = screen_w
#        self.screen_h = screen_h
#        self.camera_w = camera_w
#        self.camera_h = camera_h
#
#                
#
#    def add_calibration_point(self, input, output):
#
#    def get_target(self, input):
#
#    def hash_input(self, input):
#        return 
#
#    def compute_distance(input, target):
#
#    def estimate_position(self, input):
#        x, y = self.camera_w-input.x, input.y
#
#        # Offset by the center of the camera centers
#        camera_center_x = (input.tr.x + input.br.x + input.bl.x + input.tl.x) / 4
#        camera_center_y = (input.tr.y + input.br.y + input.bl.y + input.tl.y) / 4
#        x -= camera_center_x
#        y -= camera_center_y
#
#        # Offset by the angle of the camera

class AI_Calibrate(torch.nn.Module):

    def __init__(self, screen_w, screen_h, camera_w, camera_h):
        super(AI_Calibrate, self).__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.camera_w = camera_w
        self.camera_h = camera_h


        self.screen_dims = torch.tensor([screen_w, screen_h]).reshape(1, 2)
        self.camera_dims = torch.tensor([camera_w, camera_h]).reshape(2, 1)

        self.layer1 = torch.nn.Linear((108+10) * 2, 50)
        self.layer2 = torch.nn.Linear(50, 2)

    def forward(self, input):
        # Convert input into a tensor
        # Concatenate inputs and logged inputs
        input = input.reshape(1, 2, -1)
        input = input / self.camera_dims
        input = input.reshape(1, -1)
        logged_inputs = torch.log(input)
        input = torch.cat((logged_inputs, input), 1)
        

        input = self.layer1(input)
        input = torch.nn.Tanh()(input)
        input = self.layer2(input)
        input = torch.nn.Tanh()(input)
        
        input = input*self.screen_dims

        return input

class Predictor():
    def __init__(self, screen_w, screen_h, camera_w, camera_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.camera_w = camera_w
        self.camera_h = camera_h

        self.model = AI_Calibrate(screen_w, screen_h, camera_w, camera_h)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def add_calibration_point(self, input, output):
        input = input.convert_to_torch()
        output = output.convert_to_torch().reshape(1,2)
        self.optim.zero_grad()
        pred_output = self.model(input)
        loss = torch.nn.MSELoss()(pred_output, output)
        loss.backward()
        self.optim.step()

    def __call__(self, input):
        input = input.convert_to_torch()
        output = self.model(input)
        return Point(output[0,0], output[0,1])

class CoordinatePoint():
    def convert_to_torch(self):
        pass

class Input():
    def __init__(self, pupil=None, left_pupil_world=None, right_pupil_world=None, buffer=None):

        # Code not used currently
        if buffer is None:
            self.left_pupil_world = left_pupil_world
            self.right_pupil_world = right_pupil_world
            self.pupil = pupil
        else:
            try:
                self.markers = pickle.loads(buffer[0])
                #self.right_pupil_world = pickle.loads(buffer[1])
                #iris = np.zeros((Iris.nums), dtype=np.float32)
                self.pupil = pickle.loads(buffer[1])
                self.initialized = True
            except:
                self.initialized = False

    def convert_to_torch(self):
        # Concatenate the pupil, left_pupil_world, and right_pupil_world into a tensor
        pupil_as_tensor = self.pupil.convert_to_torch()
        markers_as_tensor = self.markers.convert_to_torch()
        #left_pupil_world_as_tensor = self.left_pupil_world.convert_to_torch()
        #right_pupil_world_as_tensor = self.right_pupil_world.convert_to_torch()
        return torch.cat((pupil_as_tensor, markers_as_tensor), 0)


class Point():
    bytes = np.zeros((2), dtype=np.float32).nbytes
    nums = 2
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point(x=%f, y=%f)" % (self.x, self.y)

    def convert_to_torch(self):
        return torch.tensor([self.x, self.y], dtype=torch.float32)
    
    def move_to_shared_mem(self, buffer, index):
        buffer[index] = pickle.dumps(self)


class Iris():
    
    bytes = np.zeros((8), dtype=np.float32).nbytes
    nums = 10
    def __init__(self, eye, iris):
        self.eye = eye
        self.iris = iris[:,:,:2].flatten()
    
    def __repr__(self):
        return "Iris(x=%f, y=%f)" % (self.iris[0], self.iris[1])

    def convert_to_torch(self):
        return self.iris
    def move_to_shared_mem(self, buffer, index):
        buffer[index] = pickle.dumps(self)

class MarkerPose():
    bytes = np.zeros((4), dtype=np.float32).nbytes
    nums = 4
    def __init__(self, markers):

        self.markers = markers

    def __repr__(self):
        return "MarkerPose(x=%f, y=%f)" % (self.markers[0], self.markers[1])

    def convert_to_torch(self):
        return torch.from_numpy(self.markers).flatten()
    
    def move_to_shared_mem(self, buffer, index):
        buffer[index] = pickle.dumps(self)

class ChessBoard():
    bytes = np.zeros((4), dtype=np.float32).nbytes
    nums = 54*2

    def __init__(self, markers):
        self.markers = markers

    def __repr__(self):
        return "ChessBoard(x=%f, y=%f)" % (self.markers[0], self.markers[1])

    def convert_to_torch(self):
        return torch.from_numpy(self.markers).flatten()

    def move_to_shared_mem(self, buffer, index):
        buffer[index] = pickle.dumps(self)
