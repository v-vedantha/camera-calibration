import math
# We want to first find the approximate location of the pupil
import numpy as np

# Then we want to edge detect the boundaries of the pupil

# And then fit an ellipse to the boundaries, and return the ellipse parameters

import cv2

# Read the original image
# Read from input video
import sys
vid_path = sys.argv[1]
video = cv2.VideoCapture(vid_path)

for _ in range(int(sys.argv[2])):
    img = video.read()[1]


# Display original image

# Convert to graycsale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Original', img_gray)
# cv2.waitKey(0)
# # Blur the image for better edge detection

# # Sobel Edge Detection
# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# # Display Sobel Edge Detection Images
# cv2.imshow('Sobel X', sobelx)
# cv2.waitKey(0)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)

# # Canny Edge Detection
# edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# # Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
params = {
    "threshold" : 56,
    "circularity" : 1.3,
    "extend" : 0.85,
    'area_min' : 1000,
    'area_max' : 6000,
    'left_boundary' : 100,
    'right_boundary' : 1000,
    'top_boundary' : 100,
    'bottom_boundary' : 1000,
    'previous_ellipse' : None
}
import time
image = img
start = time.time()
# Since each iteration takes around 0.01 seconds, and for real time we only have at most 10 iterations
# We can grid search for the correct parameters
# Then save those and search in that region for the next frame and so on so forth
# It's a best effort solution which might work
# The other technique is to estimate the location of the pupil using ML
# THen we can try and calculate the parameters needed to guess it better
# But because you are detecting the pupil and not hte iris with the infrared
# We know the circle has to be mostly complete minus some white dots from the light
# To get rid of the white dots, we can for example blur the image
# Or we can try to 

# Even better we can use the result of the previous one to check that the
# Bounding threshold is corect. I.e. check if you can get a circle with a decent oval
# ness and area at the bounding threshold and so on
plot = True

def pupil_process(buffer, index_start, cam_index, mapper):
    
    camera = cv2.VideoCapture(cam_index)
    previous_result = None
    while True:
        frame = camera.read()[1]
        if frame is None:
            continue
        result = detect(frame, params, previous_result=previous_result)
        if result != None:
            mapper(result).move_to_shared_mem(buffer, index_start)

def is_near(ellipse1, ellipse2):
    return (ellipse1[0][0] - ellipse2[0][0]) ** 2 + (ellipse1[0][1] - ellipse2[0][1]) ** 2 < 100

def detect(image, params, previous_result=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if plot:
        cv2.imshow("image_unblurred", gray)
        cv2.waitKey(0)
    retval, thresholded = cv2.threshold(gray, params['threshold'], 255, 0)

    for r in range(1, 6):
        if plot:
            cv2.imshow("threshold", thresholded)
            cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*r + 1, 2 * r + 1))
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    if plot:
        cv2.imshow("threshold", thresholded)
        cv2.waitKey(0)

    closed = thresholded

    if plot:
        cv2.imshow("closed", closed)
        cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    drawing = np.copy(image)
    cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

    # Store the number of contours that pass each condition
    num_area_pass = 0
    num_circularity_pass = 0
    num_extend_pass = 0

    for contour in contours:
        contour = cv2.convexHull(contour)

        area = cv2.contourArea(contour)
        print("Area: ", area)
        if area < params['area_min'] or area > params['area_max']:
            continue
        print("Got past area")

        num_area_pass += 1
        circumference = cv2.arcLength(contour,True)
        circularity = circumference ** 2 / (4*math.pi*area)

        print("Circularity: ", circularity)
        if circularity > params['circularity']:
            continue
        num_circularity_pass += 1
        print("Got past circularity")

        bounding_box = cv2.boundingRect(contour)

        extend = area / (bounding_box[2] * bounding_box[3])

        # reject the contours with big extend
        print("Extend: ", extend)
        if extend > params['extend']:
            continue
        print("Got past extend")
        num_extend_pass += 1

        # calculate countour center and draw a dot there
        m = cv2.moments(contour)
        if m['m00'] != 0:
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            cv2.circle(drawing, center, 3, (0, 255, 0), -1)

        # fit an ellipse around the contour and draw it into the image

        print(circularity, area)
        ellipse = cv2.fitEllipse(contour)
        if params['previous_ellipse'] is not None:
            if not is_near(params['previous_ellipse'], ellipse):
                continue

        params['previous_ellipse'] = ellipse
        print(ellipse)
        cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))
        if True:
            cv2.imshow("Drawing", drawing)
            cv2.waitKey(0)
        return ellipse
    
    # If you do find the eye, then update the darkness threshold
    #       if num_extend_pass == 1:



def get_average_values(image):
    ellipse_parameters = params['previous_ellipse']
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(image.shape[0], image.shape[1], 1)
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    x, y = np.meshgrid(x, y)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    coordinates = np.concatenate((y, x), axis=2)

    center1 = np.array([ellipse_parameters[0][1], ellipse_parameters[0][0]])

    d = np.sqrt((np.sum((coordinates - center1) ** 2, axis=2).reshape(image.shape[0], image.shape[1], 1)))

    a = np.zeros_like(image)
    a[d < sum(ellipse_parameters[1])/4.2] = 200
    cv2.imshow("close", a)
    cv2.waitKey(0)

    # Average pixel in ellipse is
    return np.mean(image[d < sum(ellipse_parameters[1])/4.2])

# show the frame
#detect(img, params)

for _ in range(1000):
    img = video.read()[1]
    detect(img, params)
    params['threshold'] = get_average_values(img)
    print(params['threshold'], 'new threshold')

print(time.time() - start)



