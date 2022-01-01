# import the pygame module, so you can use it
import os
from eyes import FrameProcesser
import pickle
import pygame
import numpy as np
import cv2
import pyautogui
 
# define a main function
def main():
    fp = FrameProcesser()

    cam0 = cv2.VideoCapture(0)
    #cam1 = cv2.VideoCapture(1)
    cameras = [cam0]#, cam1]

    # initialize the pygame module
    images = []
    for i in range(len(cameras)):
        index = i
        retval, image = cameras[i].read()
        images.append(image)
        cv2.imwrite(f"calibration_image_{i}.jpg", image)
    breakpoint()

    pygame.init()
    # create a surface on screen that has the size of 240 x 180

    width, height = pyautogui.size().width, pyautogui.size().height
    screen = pygame.display.set_mode((width,height))
     
    # define a variable to control the main loop
    running = True
    index = 0     
    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():

            if event.type == pygame.MOUSEBUTTONDOWN:
                index += 1
                # Take a picture using both cameras
                images = []
                if not os.path.exists(f"calibration_images/{index}"):
                    os.makedirs(f"calibration_images/{index}")
                if not os.path.exists(f"mouse_positions/{index}"):
                    os.makedirs(f"mouse_positions/{index}")

                for i in range(len(cameras)):

                    retval, image = cameras[i].read()
                    images.append(image)
                    cv2.imwrite(f"calibration_images/{index}/calibration_image_{i}.jpg", image)
                    fp.process(image)
                                
     
                # Get the mouse position from pygame
                (x, y) = pygame.mouse.get_pos()
                with open(f"mouse_positions/{index}/positions.pkl", "wb") as f:
                    pickle.dump((x, y), f)

                print(x, y)
     
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
