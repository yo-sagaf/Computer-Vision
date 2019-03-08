import numpy as np
import os
import cv2 as cv

MAX_ITERATIONS = 1000  # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
EPSILON = 0.002  # the stopping criterion for the difference when performing the Horn-Schuck algorithm
EIGEN_THRESHOLD = 0.01  # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25, 'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    # the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    return flow


# ***********************************************************************************
# implement Lucas-Kanade Optical Flow
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Lucas-Kanade algorithm
def Lucas_Kanade_flow(frames, Ix, Iy, It, window_size):
    # Pad the matrices with 0 to fit a 15x15 matrix for all pixels
    # Slide 25
#     IxIx = np.pad(np.square(Ix), window_size[0]-1, mode='constant', constant_values=0)
#     IyIy = np.pad(np.square(Iy), window_size[0]-1, mode='constant', constant_values=0)
#     IxIy = np.pad(Ix * Iy, window_size[0]-1, mode='constant', constant_values=0)
#     IxIt = np.pad(Ix * It, window_size[0]-1, mode='constant', constant_values=0)
#     IyIt = np.pad(Iy * It, window_size[0]-1, mode='constant', constant_values=0)
    IxIx = np.square(Ix)
    IyIy = np.square(Iy)
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It

    estimated_flow = np.zeros((Ix.shape[0], Ix.shape[1], 2))
    for y in range(window_size[0]-1, Ix.shape[0]-window_size[0]):
        for x in range(window_size[0]-1, Ix.shape[1]-window_size[0]):
                winIxIx = np.sum(IxIx[y-14:y+14, x-14:x+14])
                winIyIy = np.sum(IyIy[y-14:y+14, x-14:x+14])
                winIxIy = np.sum(IxIy[y-14:y+14, x-14:x+14])
                winIxIt = np.sum(IxIt[y-14:y+14, x-14:x+14])
                winIyIt = np.sum(IyIt[y-14:y+14, x-14:x+14])
                M = np.array([[winIxIx, winIxIy], [winIxIy, winIyIy]])
                b = np.array([[winIxIt], [winIyIt]]) * -1
                invM = np.linalg.inv(M)
                uv = np.matmul(invM, b)
                estimated_flow[y, x, :] = uv.reshape(-1)

    return estimated_flow


# ***********************************************************************************
# implement Horn-Schunck Optical Flow
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Horn-Schunck algorithm
def Horn_Schunck_flow(Ix, Iy, It):
    i = 0
    diff = 1
    alpha = np.ones((Ix.shape[0], Ix.shape[1]))
    # initialise u, v
    delta_u = np.zeros((Ix.shape[0], Ix.shape[1]))
    delta_v = np.zeros((Ix.shape[0], Ix.shape[1]))
    u_prev = np.zeros((Ix.shape[0], Ix.shape[1]))
    v_prev = np.zeros((Ix.shape[0], Ix.shape[1]))
    estimated_flow = np.zeros((Ix.shape[0], Ix.shape[1], 2))
    while i < MAX_ITERATIONS and diff > EPSILON:  # Iterate until the max number of iterations is reached or the difference is less than epsilon
        delta_u = cv.Laplacian(src=u_prev, ddepth=-1, ksize=1, scale=0.25)
        delta_v = cv.Laplacian(src=v_prev, ddepth=-1, ksize=1, scale=0.25)
        u_prev += delta_u
        v_prev += delta_v
        Ixx = np.square(Ix)
        Ixy = Ix * Iy
        Iyy = np.square(Iy)

        temp_u = Ixx * u_prev + Ixy * v_prev + Ix * It
        denominator = np.square(alpha) + Ixx + Iyy
        temp_u /= denominator
        u_new = u_prev - temp_u

        temp_v = Ixy * u_prev + Iyy * v_prev + Iy * It
        temp_v /= denominator
        v_new = v_prev - temp_v

        diff_u = u_new - u_prev
        diff_v = v_new - v_prev
        diff = cv.norm(diff_u, cv.NORM_L1) + cv.norm(diff_v, cv.NORM_L1)

        u_prev = u_new
        v_prev = v_new

        i += 1

    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            estimated_flow[i, j, 0] = u_new[i, j]
            estimated_flow[i, j, 1] = v_new[i, j]
    return estimated_flow


# calculate the angular error here
def calculate_angular_error(estimated_flow, groundtruth_flow):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/
    error = np.sum(estimated_flow * groundtruth_flow, axis=2)
    return np.mean(error)


# function for converting flow map to to BGR image for visualisation
def flow_map_to_bgr(flow_map):
    # https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    hsv = np.zeros((388, 584, 3), dtype=np.uint8)
    mag, ang = cv.cartToPolar(flow_map[:, :, 0], flow_map[:, :, 1])
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


if __name__ == "__main__":
    # read your data here and then call the different algorithms, then visualise your results
    WINDOW_SIZE = [15, 15]  # the number of points taken in the neighborhood of each pixel when applying Lucas-Kanade
    gt_flow = load_FLO_file('../data/groundTruthOF.flo')
    imgFrame1 = cv.imread('../data/frame1.png')
    imgFrame1Gray = cv.cvtColor(imgFrame1, cv.COLOR_BGR2GRAY).astype('float64')
    imgFrame1Gray /= 255
    imgFrame2 = cv.imread('../data/frame2.png')
    imgFrame2Gray = cv.cvtColor(imgFrame2, cv.COLOR_BGR2GRAY).astype('float64')
    imgFrame2Gray /= 255

    # Gradient along x direction
    Ix = cv.Sobel(imgFrame1Gray, -1, 1, 0, ksize=3)
    # Gradient along y direction
    Iy = cv.Sobel(imgFrame1Gray, -1, 0, 1, ksize=3)
    # Temporal difference
    It = imgFrame1Gray - imgFrame2Gray

    estimated_flow = Lucas_Kanade_flow([imgFrame1Gray, imgFrame2Gray], Ix, Iy, It, WINDOW_SIZE)    
    print('Average angular error for Lucas Kanade Flow = {}'.format(calculate_angular_error(estimated_flow, gt_flow)))
    print('--------------------------------------------------')
    cv.imshow("Ground Truth flow - Lucas Kanade ", flow_map_to_bgr(gt_flow))
    cv.imshow("Estimated flow - Lucas Kanade ", flow_map_to_bgr(estimated_flow))
    cv.waitKey(0)
    cv.destroyAllWindows()

    estimated_flow_hs = Horn_Schunck_flow(Ix, Iy, It)
    print('Average angular error for Horn Schunk = {}'.format(calculate_angular_error(estimated_flow_hs, gt_flow)))
    print('--------------------------------------------------')
    cv.imshow("Ground Truth flow - Horn Schunk", flow_map_to_bgr(gt_flow))
    cv.imshow("Estimated flow - Horn Schunk", flow_map_to_bgr(estimated_flow_hs))
    cv.waitKey(0)
    cv.destroyAllWindows()