from eyeloop.extractors.custom import custom_Extractor
from calibration_dataset import CalibrationDataset
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory, ShareableList
import os
from find_eye_center import histogram, detect
from coordinate_mapping import AI_Calibrate, Point, Iris, Input, MarkerPose, ChessBoard, Ellipse
import random
import torch
from tqdm import tqdm
import cv2
from pynput import mouse
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

all_files = os.listdir('calibration_images')

train_files = []
test_files = []

for file in all_files:
    if random.random() < 0.8:
        train_files.append(file)
    else:
        test_files.append(file)

# datasets
train_dataset = CalibrationDataset(train_files)
test_dataset = CalibrationDataset(test_files)

# dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


nline = 6
ncol = 9
buffer = ShareableList(['-'*10000, '-'*10000])
failed = 0
detect_missed = 0

avg = torch.tensor([1058.1882,  519.2062, 1063.2218,  549.5897, 1067.9385,  580.3976,
        1072.9686,  610.8581, 1077.9069,  641.6965, 1083.0876,  672.4626,
        1027.7004,  523.9083, 1032.7506,  554.6132, 1037.5879,  585.2173,
        1042.2834,  616.1381, 1047.3152,  646.4442, 1052.1624,  677.7376,
         996.9419,  529.4602, 1002.1187,  559.3801, 1006.4456,  590.4249,
        1011.8125,  620.9095, 1016.3878,  651.6868, 1021.4254,  682.3707,
         967.0753,  534.2369,  971.3307,  564.7801,  976.2697,  594.8834,
         980.9216,  625.8707,  985.8541,  656.1960,  990.6652,  687.3126,
         936.9162,  539.4611,  941.3528,  569.5844,  945.5202,  600.0405,
         950.5626,  630.6069,  955.0010,  661.1625,  960.2617,  691.8041,
         907.8370,  544.6057,  911.2488,  574.6255,  915.8574,  604.6323,
         920.0760,  635.4027,  924.9362,  665.7038,  929.8605,  696.5096,
         878.6294,  550.1141,  882.0646,  579.4238,  885.8624,  609.5617,
         890.3942,  639.7729,  894.8342,  670.3541,  899.6299,  700.9897,
         851.2900,  555.9808,  853.4678,  585.1346,  856.9489,  614.5173,
         860.8380,  644.5652,  865.2167,  674.8307,  870.0304,  705.2320,
         824.6730,  561.9899,  827.5447,  590.6631,  828.9716,  619.7949,
         831.9328,  649.2489,  835.9683,  679.3134,  840.6730,  709.6014,
         959.0427,  337.0685,  137.8030,  148.5260,   79.8299])
std = torch.tensor([32.6572, 25.8865, 32.3793, 25.5486, 32.4619, 25.4114, 32.0149, 25.0784,
        31.6864, 24.9735, 31.2348, 24.8101, 33.0007, 26.0213, 32.6380, 25.7461,
        32.3097, 25.4907, 32.2240, 25.3263, 31.9285, 25.0100, 31.7345, 24.9666,
        33.1284, 26.3174, 32.9485, 25.9610, 32.7845, 25.8628, 32.3955, 25.5781,
        32.1612, 25.3952, 31.7561, 25.1851, 33.5338, 26.5331, 33.1176, 26.3714,
        32.7410, 25.7102, 32.7781, 25.8379, 32.5660, 25.6331, 32.2371, 25.4016,
        33.7710, 26.5269, 33.3030, 26.2986, 33.2149, 26.0561, 32.9791, 25.9922,
        32.6417, 25.5091, 32.5083, 25.3637, 33.6684, 26.6852, 33.7358, 26.4797,
        33.2337, 26.3066, 33.0400, 25.9908, 32.6414, 25.9796, 32.6645, 25.7190,
        33.8565, 26.8324, 33.9147, 26.7025, 33.6308, 26.3631, 33.2979, 26.1850,
        33.2543, 25.9369, 33.0928, 25.8054, 34.2555, 27.1107, 34.0940, 26.9620,
        33.9607, 26.6277, 33.6571, 26.3607, 33.3076, 26.0375, 33.0711, 25.9399,
        34.3710, 27.2444, 34.4338, 26.8473, 34.1103, 26.7613, 33.7361, 26.5049,
        33.6742, 26.3900, 33.4425, 25.9415, 99.0152, 57.7941, 12.9714, 20.4164,
        49.6819])

p_mean = torch.tensor([700.8484, 415.9319])
p_std = torch.tensor([387.3046, 234.4409])
writer = SummaryWriter()

model = AI_Calibrate()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
for epoch in range(100):
    avg_loss = []
    test_loss = []
    for i, (input, position) in enumerate(tqdm(train_loader)):
        input = ((input - avg)/std)
        position = ((position - p_mean)) / p_std
        optim.zero_grad()
        output = model(input)
        loss = criterion(output, position)
        avg_loss.append(loss.item())
        loss.backward()
        optim.step()

    for i, (input, position) in enumerate(tqdm(test_loader)):
        input = ((input - avg)/std)
        position = ((position - p_mean)) / p_std
        output = model(input)
        loss = criterion(output, position)
        test_loss.append(loss.item())
    writer.add_scalar('loss', sum(avg_loss)/ len(avg_loss), epoch)
    writer.add_scalar('test_loss', sum(test_loss)/ len(test_loss), epoch)
        
