import cv2
import numpy as np
import random
np.random.seed(42)

#   =======================================================
#                   Task1
#   =======================================================

img = cv2.imread('../images/building.jpeg')
img1 = cv2.imread('../images/building.jpeg', 0).astype('float64')
img1 /= 255

Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
Ix2 = cv2.GaussianBlur((Ix ** 2), (0, 0), 2)
Iy2 = cv2.GaussianBlur((Iy ** 2), (0, 0), 2)
Ixy = cv2.GaussianBlur((Ix * Iy), (0, 0), 2)

k = 0.04
response_max = 0
response_func = np.zeros((img1.shape[0], img1.shape[1]))
for p in range(img1.shape[0]):
    for q in range(img1.shape[1]):
        # compute structural tensor
        M = np.array([[Ix2[p, q], Ixy[p, q]], [Ixy[p, q], Iy2[p, q]]])
        response_func[p, q] = np.linalg.det(M) - k * (np.trace(M) ** 2)
        if response_max < response_func[p, q]:
            response_max = response_func[p, q]

threshold = 0.03 * response_max
detected_corners = np.zeros((img1.shape[0], img1.shape[1]))
#Harris Corner Detection
outImg = img.copy()
for p in range(img1.shape[0] - 1):
    for q in range(img1.shape[1] - 1):
        if response_func[p, q] > threshold:
            detected_corners[p, q] = 1  # corner
            cv2.circle(outImg, (q,p), 1, (255,0,0), thickness=1)

cv2.imshow('Response function', response_func)
cv2.imshow('Harris Corners Threshold', detected_corners)
cv2.imshow('Harris Corners Detector', outImg)

#Forstner Corner Detection
w = np.zeros(img1.shape)
qt = np.zeros(img1.shape)
for p in range(img1.shape[0]):
    for q in range(img1.shape[1]):
        # compute structural tensor
        M = np.array([[Ix2[p, q], Ixy[p, q]], [Ixy[p, q], Iy2[p, q]]])
        determinant_M = np.linalg.det(M)
        trace_M = np.trace(M)
        w[p, q] = determinant_M / trace_M
        qt[p, q] = 4 * determinant_M / (trace_M ** 2)

wmin = np.min(w)
qmin = np.min(qt)


outImg = img.copy()
detected_corners1 = np.zeros(img1.shape)
#Harris Corner Detection
for p in range(img1.shape[0]):
    for q in range(img1.shape[1]):
        if img1[p, q] < wmin and img1[p, q] < qmin:
            detected_corners1[p, q] = 1  # corner
            cv2.circle(outImg, (q,p), 1, (255,0,0), thickness=1)

cv2.imshow('w', w)
cv2.imshow('q', qt)
cv2.imshow('Foerstner Corners Threshold', detected_corners1)
cv2.imshow('Foerstner Corners Detector', outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()




#   =======================================================
#                   Task2
#   =======================================================

img1 = cv2.imread('../images/mountain1.png')
img2 = cv2.imread('../images/mountain2.png')

#extract sift keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# own implementation of matching

good_matches = []

for idx in range(len(des1)):
    distance = np.sum(np.square(des1[idx] - des2), axis=1)
    bestMatches = (distance.argsort()[:2])
    if distance[bestMatches[0]] < 0.4 * distance[bestMatches[1]]:
        dMatch = cv2.DMatch()
        dMatch.distance = distance[bestMatches[0]]
        dMatch.imgIdx = 0
        dMatch.trainIdx = int(bestMatches[0])
        dMatch.queryIdx = idx
        good_matches.append([dMatch])

img=None
img=cv2.drawKeypoints(img1,kp1,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Keypoints 1', img)
img=cv2.drawKeypoints(img2,kp2,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Keypoints 2', img)

# display matched keypoints
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imshow('Matched Keypoints', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  =======================================================
#                          Task-3                         
#  =======================================================

nSamples = 4
nIterations = 20
thresh = 0.1
minSamples = 4
bestTransform = None
bestInliers = 0

#  /// RANSAC loop
for i in range(nIterations):

    print('iteration '+str(i))
    
    #randomly select 4 pairs of keypoints
    pts1 = []
    pts2 = []
    randMatches = np.random.choice(np.arange(len(good_matches)), 4)
    for j in randMatches:
        pt1 = good_matches[j][0].queryIdx
        pt2 = good_matches[j][0].trainIdx
        pts1.append(kp1[pt1].pt)
        pts2.append(kp2[pt2].pt)
    pts1 = np.array(pts1).astype('float32')
    pts2 = np.array(pts2).astype('float32')

    #compute transofrmation and warp img2 using it
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transImg = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    
    #count inliers and keep transformation if it is better than the best so far
    inliers = 0

    for idx in range(len(kp1)):
        size = int(kp1[idx].size/2)
        if size > 0:
            x, y = int(kp1[idx].pt[0]), int(kp1[idx].pt[1])
            # Extract feature vectors around keypoint
            feature1 = img1[y-size:y+size, x-size:x+size].reshape(-1,1)
            feature2 = transImg[y-size:y+size, x-size:x+size].reshape(-1,1)
            # Calculate MSE and normalize
            error = np.sum(np.square(feature1-feature2))/(feature1.size*3*255)
            if error < thresh:
                inliers += 1
        
    if inliers > bestInliers and inliers > minSamples:
        bestTransform = M
        bestInliers = inliers


#apply best transformation to transform img2 
transImg = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
cv2.imshow('Warped Image', transImg)

#display stitched images
# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
stitchedImg = cv2.addWeighted(img1, 0.8, transImg, 0.05, 0)
cv2.imshow('Stitched Image', stitchedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()