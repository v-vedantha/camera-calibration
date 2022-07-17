# import the pygame module, so you can use it
import os
import pickle
import pygame
import numpy as np
import cv2
import pyautogui
 
import random
# define a main function
def main():

    clock = pygame.time.Clock()
    cam0 = cv2.VideoCapture(0)
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)
    cameras = [cam0, cam1, cam2]

    # initialize the pygame module
    images = []
    for i in range(len(cameras)):
        index = i
        retval, image = cameras[i].read()
        images.append(image)
        cv2.imwrite(f"calibration_image_{i}.jpg", image)

    pygame.init()
    # create a surface on screen that has the size of 240 x 180

    width, height = pyautogui.size().width, pyautogui.size().height
     
    # define a variable to control the main loop
    running = True
    index = 0     
    # main loop
    while running:
        screen = pygame.display.set_mode((width,height))
        # event handling, gets all event from the event queue
        x = random.random() * 1440
        y = random.random() * 900
        # Draw a circle at x, y
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 15)
        pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 10)
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 3)
        # update the screen
        pygame.display.flip()

        clock.tick(.7)
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

        with open(f"mouse_positions/{index}/positions.pkl", "wb") as f:
            pickle.dump((x, y), f)

        print(x, y)
     
            # only do something if the event is of type QUIT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
