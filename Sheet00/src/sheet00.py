import cv2 as cv
import numpy as np
import random
import math
import sys
import os
import time
random.seed(time.time)

if __name__ == '__main__':
    img_path = sys.argv[1]
    fileName = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/' + img_path
    print(fileName)

    # 2a: read and display the image
    # img = cv.imread('../images/bonn.png')
    img = cv.imread(fileName)
    cv.imshow('Original Image', img)
    cv.waitKey(0)

    # 2b: display the intenstity image
    intensityImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Intensity Image', intensityImg)
    cv.waitKey(0)

    # 2c: for loop to perform the operation
    diffImg = img.copy()

    for y in range(intensityImg.shape[0]):
        for x in range(intensityImg.shape[1]):
            diffImg[y, x, :] -= np.uint8(intensityImg[y, x] * 0.5)
        
    diffImg[diffImg < 0] = 0
    cv.imshow('Difference Image', diffImg)
    cv.waitKey(0)

    # 2d: one-line statement to perfom the operation above
    diffImg2 = img - np.uint8(np.expand_dims(intensityImg, axis = 2) * 0.5)
    diffImg2[diffImg2 < 0] = 0
    cv.imshow('Difference Image 2', diffImg2)
    cv.waitKey(0)


    # 2e: Extract a random patch
    cy = int(img.shape[0]/2)
    cx = int(img.shape[1]/2)
    imgPatch = img[cy - 8 : cy + 8, cx - 8 : cx + 8]
    cv.imshow('Image Patch', imgPatch)
    cv.waitKey(0)

    ry = random.randint(0,img.shape[0]-16)
    rx = random.randint(0,img.shape[1]-16)
    imgWithPatch = img.copy()
    imgWithPatch[ry : ry + 16, rx : rx + 16] = imgPatch[:,:]
    cv.imshow('Image with Patch', imgWithPatch)
    cv.waitKey(0)

    # 2f: Draw random rectangles and ellipses
    imgRectangles = img.copy()
    for i in range(10):
        ry = random.randint(0,img.shape[0])
        rx = random.randint(0,img.shape[1])
        w = random.randint(1,50)
        h = random.randint(1,50)
        cv.rectangle(imgRectangles, (ry, rx), (ry + w, rx + x), (255, 0, 0), 2)
    
    cv.imshow('Random Rectangles', imgRectangles)
    cv.waitKey(0)


    # draw ellipses
    imgEllipses = img.copy()
    for i in range(10):
        ry = random.randint(0,img.shape[0])
        rx = random.randint(0,img.shape[1])
        raxisy = random.randint(0,150)
        raxisx = random.randint(0,150)
        cv.ellipse(imgEllipses, (ry, rx), (raxisy, raxisx), 0, 0, 360, (0, 255, 0), 3)
    
    cv.imshow('Random Ellipses', imgEllipses)
    cv.waitKey(0)


    # destroy all windows
    cv.destroyAllWindows()