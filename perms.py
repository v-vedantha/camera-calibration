from pynput.mouse import Button, Controller
from pynput import mouse as m
import pyautogui

# Move mosue to 0,0
#pyautogui.moveTo(0, 0, 1)

mouse = Controller()

# Read pointer position
print('The current pointer position is {0}'.format(
    mouse.position))

# Set pointer position
mouse.position = (10, 20)
print('Now we have moved it to {0}'.format(
    mouse.position))

# Move pointer relative to current position
mouse.move(500, -500)

# Press and release
mouse.press(Button.left)
mouse.release(Button.left)

# Double click; this is different from pressing and releasing
# twice on Mac OSX
mouse.click(Button.left, 2)

# Scroll two steps down
mouse.scroll(0, 2)

def f(x, y, b, p):
    print(x, y, b, p)

m.Listener(on_click=f).start()

import time
time.sleep(20)
