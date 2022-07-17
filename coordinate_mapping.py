import torch
import numpy as np
import pickle
import tensorboardX
from tqdm import tqdm
from calibration_dataset import SimpleDataset

class AI_Calibrate(torch.nn.Module):

    def __init__(self):
        super(AI_Calibrate, self).__init__()
        self.layer1 = torch.nn.Linear(113, 1150)
        self.layer2 = torch.nn.Linear(1150, 150)
        self.layer3 = torch.nn.Linear(150, 150)
        self.layer4 = torch.nn.Linear(150, 50)
        self.layer5 = torch.nn.Linear(50, 2)

    def forward(self, input):
        input = self.layer1(input)
        input = torch.nn.LeakyReLU()(input)
        input = self.layer2(input)
        input = torch.nn.LeakyReLU()(input)
        input = self.layer3(input)
        input = torch.nn.LeakyReLU()(input)
        input = self.layer4(input)
        input = torch.nn.LeakyReLU()(input)
        input = self.layer5(input)

        return input

class Predictor():
    def __init__(self):
        self.writer = tensorboardX.SummaryWriter('')
        self.index=  0

        self.mean_inputs = None
        self.mean_outputs = None
        self.std_inputs = None
        self.std_outputs = None

        self.model = AI_Calibrate()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
    
    def calibrate(self, inputs, outputs):
        self.mean_inputs = torch.mean(inputs, 0)
        self.mean_outputs = torch.mean(outputs, 0)
        self.std_inputs = torch.std(inputs, 0)
        self.std_outputs = torch.std(outputs, 0)

        inputs = (inputs - self.mean_inputs) / self.std_inputs
        outputs = (outputs - self.mean_outputs) / self.std_outputs

        train_dataset = SimpleDataset(inputs, outputs)
        test_dataset = SimpleDataset(inputs, outputs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)

        for epoch in range(60):
            avg_loss = []
            test_loss = []
            for input, output in tqdm(train_loader):
                avg_loss.append(self.train_single_point(input, output))

            for input, output in tqdm(test_loader):
                prediction = self.infer(input)
                loss = self.criterion(output, prediction)
                test_loss.append(loss.item())
            
            self.writer.add_scalar('loss', sum(avg_loss)/ len(avg_loss), epoch)
            self.writer.add_scalar('test_loss', sum(test_loss)/ len(test_loss), epoch)

    def infer(self, input):
        input = (input - self.mean_inputs) / self.std_inputs
        output = self.model(input)
        return (output * self.std_outputs) + self.mean_inputs

    def train_single_point(self, input, output):
        input = (input - self.mean_inputs) / self.std_inputs
        output = (output - self.mean_outputs) / self.std_outputs
        self.optim.zero_grad()
        inference = self.model(input)
        loss = self.criterion(output, inference)
        loss.backward()
        self.optim.step()
        return loss.item()

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

class Ellipse():
    bytes = np.zeros((4), dtype=np.float32).nbytes
    nums = 4

    def __init__(self, ellipse):
        self.ellipse = ellipse

    def __repr__(self):
        return "Ellipse(x=%f, y=%f)" % (self.ellipse[0], self.ellipse[1])

    def convert_to_torch(self):
        center0 = list(self.ellipse[0])
        center1 = list(self.ellipse[1])
        radius = [self.ellipse[2]]
        return torch.tensor(center0 + center1 + radius, dtype=torch.float32).flatten()

    def move_to_shared_mem(self, buffer, index):
        buffer[index] = pickle.dumps(self)
