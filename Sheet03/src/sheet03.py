import numpy as np
import cv2 as cv
import random
import time

random.seed(0)

def displayImage(winName, img):
    """ Helper function to display image
    arguments:
    winName -- Name of display window
    img     -- Source Image
    """
    cv.imshow(winName, img)
    cv.waitKey(0)

##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny( gray_image,50,150)
    #cv.imshow('edges', edges)
    detected_lines = cv.HoughLines(edges,1,np.pi/180,10)
    #print (detected_lines)
    for rho,theta in detected_lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)
    displayImage('1_a Hough transform - detected lines ', img)
    



def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    rho = int(np.linalg.norm(img_edges.shape) / d_resolution)
    #print (rho)
    theta = int(180 / theta_step_sz)
    theta_array = np.deg2rad(np.arange(-90, 90, theta_step_sz))
    #print (theta)
    width, height = img_edges.shape
    img_edges_copy = img_edges.copy()
    detected_lines = []
    for x in range(width):
        for y in range(height):
            if img_edges_copy[x,y]:
                for index_theta in range(len(theta_array)):
                    #theta_value = theta * index_theta 
                    rho_value = x*np.cos(theta_array[index_theta]) + y*np.sin(theta_array[index_theta])
                    # to avoid negative index
                    index_rho = int (rho_value + rho/2) 
                    # to avoid index overflow
                    if (index_rho >= rho) : continue
                    #print('rhoindex')
                    #print (index_rho)
                    accumulator[index_theta, index_rho] += 1
                    if accumulator[index_theta, index_rho] >= threshold:
                        detected_lines.append((theta_array[index_theta], rho_value))
    
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny( img_gray,50,150) # detect the edges
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    cv.imshow("1_b Accumulator myHoughLines", accumulator)
    #print (len(detected_lines))

    for theta,rho in detected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    displayImage('1_b Hough transform - own implementation', img)
 


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny( img_gray,50,150,apertureSize = 3) # detect the edges
    theta_res = 1 # set the resolution of theta
    d_res = 1 # set the distance resolution
    _, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    displayImage("task_2_ accumulator - mean shift", accumulator)
    #mean_shift(accumulator)


##############################################
#     Task 3        ##########################
##############################################

def myKmeans(data, k, useDist = False):
    """
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, 1), dtype = int)
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    threshold = 0
    if data.shape[1] > 1:
        threshold = 20

    print('Threshold value = ' + str(threshold))
    print('-------------------------------------------------')

    # initialize centers using some random points from data
    # ....

    # Randomly initialize centers with pixel difference of greater than 0

    for idx in range(centers.shape[0]):
        randIdx = random.choice(range(data.shape[0]))
        centers[idx] = randIdx

    # Randomly initialize centers of different pixl values. Still buggy
    # start_time = time.time()
    # indices = np.arange(0,data.shape[0]).tolist()
    # for idx in range(centers.shape[0]):
    #     if len(indices) > 0:
    #         randIdx = random.choice(indices)
    #         delIndices = np.unique(np.where((data*255).astype('uint8') == (data[randIdx]*255).astype('uint8'))).tolist()
    #         if len(delIndices) > 0:
    #             for i in range(len(delIndices)):
    #                 try:
    #                     indices.remove(delIndices[i])
    #                 except ValueError:
    #                     print('Value not found')
    #             # print('Indices removed')
    #     else:
    #         randIdx = random.choice(range(data.shape[0]))
    #     centers[idx] = randIdx        
    # end_time = time.time()
    # print('Center no' + str(idx+1) + ' added in ' + str(round(end_time - start_time,5)) + ' seconds')

    # To debug uncomment the following lines
    # Sometimes the pixel values of two cluster centroids are too close
    # Therefore, one of the clusters might end up not having any points at all
    # print('Initial centers:\n' + str(centers))
    # print('-------------------------------------------------')
    # centerVals = data[centers]
    # print('Pixel Values of initial centers:\n' + str(centerVals))
    # print('-------------------------------------------------')

    convergence = False
    iterationNo = 0
    start_time = time.time()
    while not convergence:
        # assign each point to the cluster of closest center
        # ...
        euclDist = 0
        centerVals = data[centers]
        for idx in range(data.shape[0]):
            if useDist:                
                # Since data is a vector, distance is only the difference
                # Normalize the distance to keep it between 0 and 1
                euclDist = (centers - idx) / data.shape[0]
            cost = np.square(data[idx] - centerVals) + np.square(euclDist)
            index[idx] = np.random.choice(np.where(cost == np.min(cost))[0])
            clusters[index[idx]].append(idx)
            
        # update clusters' centers and check for convergence
        # ...
        convCounter = 0
        for idx in range(centers.shape[0]):
            if (len(clusters[idx]) > 0):
                if data.shape[1] == 1:
                    meanVal = np.mean(data[clusters[idx]])
                elif data.shape[1] == 3:
                    meanVal = np.mean(data[clusters[idx]], axis = 0)
                diff = (np.abs(centerVals[idx] - meanVal)*255).astype('uint8')
                if (np.sum(diff) > threshold):
                    # indices = np.unique(np.where((data*255).astype('uint8') == (meanVal*255).astype('uint8'))[0])
                    indices = np.unique(np.where((data*255).astype('uint8') == (meanVal*255).astype('uint8'))[0])
                    if indices.size > 0:
                        centers[idx] = np.random.choice(indices)
                    else:
                        # if no pixel with the mean value is found, choose another pixel in the cluster
                        # and continue
                        centers[idx] = np.random.choice(clusters[idx])
                else:
                    convCounter += 1
            else:
                convCounter += 1

        if convCounter == k:
            convergence = True
        
        iterationNo += 1
        print('iterationNo = ', iterationNo)
    
    print('-------------------------------------------------')
    end_time = time.time()
    print('Data Clustered for K = ' + str(k) + ' in ' + str(round(end_time - start_time, 5)) + ' seconds')
    print('-------------------------------------------------')

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    print('-------------------------------------------------')
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('float32')
    grayImg /= 255
    cv.imshow('Intensity Image', grayImg)
    
    K = [2, 4, 6]
    for k in K:
        print('K = ' + str(k))
        print('-------------------------------------------------')

        grayVec = np.reshape(grayImg.copy(), (-1,1))

        index, centers = myKmeans(grayVec, k)

        for kVal in range(k):
            indices = np.where(index == kVal)[0]
            grayVec[indices] = grayVec[centers[kVal]]

        cv.imshow('Segmented Intensity Image for k = ' + str(k), grayVec.reshape(grayImg.shape))

    cv.waitKey(0)
    print('=================================================')

def task_3_b():
    print("Task 3 (b) ...")
    print('-------------------------------------------------')
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    imgFloat = img.copy().astype('float64')
    imgFloat /= 255

    cv.imshow('Color Image', imgFloat)

    K = [2, 4, 6]

    for k in K:
        print('K = ' + str(k))
        print('-------------------------------------------------')

        imgVec = np.reshape(imgFloat.copy(), (-1,3))

        index, centers = myKmeans(imgVec, k)

        for kVal in range(k):
            indices = np.where(index == kVal)[0]
            imgVec[indices] = imgVec[centers[kVal]]

        cv.imshow('Segmented Color Image for k = ' + str(k), imgVec.reshape(imgFloat.shape))
    
    cv.waitKey(0)
    print('=================================================')

def task_3_c():
    print("Task 3 (c) ...")
    print('-------------------------------------------------')
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('float32')
    grayImg /= 255
    cv.imshow('Intensity Image', grayImg)
    
    K = [2, 4, 6]
    for k in K:
        print('K = ' + str(k))
        print('-------------------------------------------------')
        grayVec = np.reshape(grayImg.copy(), (-1,1))

        index, centers = myKmeans(grayVec, k, useDist = True)

        for kVal in range(k):
            indices = np.where(index == kVal)[0]
            grayVec[indices] = grayVec[centers[kVal]]

        cv.imshow('Segmented Intensity Image (Scaled Distance) for k = ' + str(k), grayVec.reshape(grayImg.shape))
    
    cv.waitKey(0)

    print('=================================================')


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    print('-------------------------------------------------')
    D = np.zeros((8,8))  
    W = np.array((
        [0, 1, 0.2, 1, 0, 0, 0, 0], # A
        [1, 0, 0.1, 0, 1, 0, 0, 0], # B
        [0.2, 0.1, 0, 1, 0, 1, 0.3, 0], # C
        [1, 0, 1, 0, 0, 1, 0, 0], # D
        [0, 1, 0, 0, 0, 0, 1, 1], # E
        [0, 0, 1, 1, 0, 0, 1, 0], # F
        [0, 0, 0.3, 0, 1, 1, 0, 1], # G
        [0, 0, 0, 0, 1, 0, 1, 0] # H
        ))  # construct the W matrix

    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:]) # construct the D matrix

    '''
    ...
    your code ...
    ...
    '''
    invSqrtD = np.linalg.inv(np.sqrt(D))
    L = D - W

    op = np.matmul(np.matmul(invSqrtD,L),invSqrtD)
    _, _, eigenVecs = cv.eigen(op)
    secMinEigenVec = eigenVecs[eigenVecs.shape[1]-2, :]

    C1 = 0
    C2 = 0
    for i in range(secMinEigenVec.shape[0]):
        if secMinEigenVec[i] < 0:
            C1 += D[i,i]
        else:
            C2 += D[i,i]

    print('Eigen Vec: ' + str(np.round(secMinEigenVec, 3)))

    # Figure in pdf
    minNormCut = (1/C1 + 1/C2) * 2.4
    print('Min Norm Cut = ' + str(minNormCut))
    print('=================================================')

##############################################
##############################################
##############################################


# task_1_a()
# task_1_b()
# task_2()
# task_3_a()
# cv.destroyAllWindows()
# task_3_b()
# cv.destroyAllWindows()
# task_3_c()
# cv.destroyAllWindows()
task_4_a()