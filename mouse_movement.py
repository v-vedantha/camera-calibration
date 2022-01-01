import pyautogui
print(pyautogui.size())
print(pyautogui.position())

# Move the mouse to 100,100
for i in range(10):
    pyautogui.moveTo(100*i, 100, duration=1)
print(pyautogui.position())
