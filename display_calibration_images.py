
# import the pygame module, so you can use it
import os
import pickle
import pygame
import numpy as np
 
# define a main function
def main():

    # initialize the pygame module
    pygame.init()
    
    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((1920,1080))
    clock=pygame.time.Clock() 
    # define a variable to control the main loop
    running = True
    index = 0
    images = []
    for i in range(1,11):
        images.append(pygame.image.load('calibration_images/'+str(i) + '/calibration_image_0.jpg'))

    # main loop
    while running:


        screen.blit(images[index], (0,0))
        
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # Get the pressed numeric keyboard value
            keys = pygame.key.get_pressed()

            for i in range(10):
                print(eval("keys[pygame.K_" + str(i) + "]"))
                if eval("keys[pygame.K_" + str(i) + "]"):
                    index = i
                    print(index)
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
        pygame.display.update()
        clock.tick(60)
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
