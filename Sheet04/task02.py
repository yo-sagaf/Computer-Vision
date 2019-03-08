import numpy as np
import cv2
import matplotlib.pyplot as plt
# from matplotlib import rc
# if you do not have latex installed simply uncomment this line + line 75
# rc('text', usetex=True)
import time

# required for the overflow errors during gradient calculations
np.seterr(over='raise')


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------


def meanCurvature(phi, epsilon):
    ''' Helper function to calculate mean Curvature term
    arguments:
    phi     -- the phi matrix containing the distance transform from the contour
    epsilon -- constant
    returns:
    dtPhi   -- matrix of change in phi values
    '''
    dtPhi = np.zeros(phi.shape)

    for y in range(1, phi.shape[0]):
        for x in range(1, phi.shape[1]):
            # equations from slide 75
            # Handle border pixels
            xMinus1 = max(0, x-1)
            yMinus1 = max(0, y-1)
            xPlus1 = min(phi.shape[1]-1, x+1)
            yPlus1 = min(phi.shape[0]-1, y+1)

            phiX = 0.5 * (phi[y, xPlus1] - phi[y, xMinus1])
            phiY = 0.5 * (phi[yPlus1, x] - phi[yMinus1, x])
            phiXX = phi[y, xPlus1] - 2 * phi[y, x] + phi[y, xMinus1]
            phiYY = phi[yPlus1, x] - 2 * phi[y, x] + phi[yMinus1, x]
            phiXY = 0.25 * (phi[yPlus1, xPlus1] - phi[yMinus1, xPlus1] -
                            phi[yPlus1, xMinus1] + phi[yMinus1, xMinus1])

            # Handle overflows in scalar values
            try:
                num = phiXX * phiY * phiY - 2 * phiX * phiY * phiXY + phiYY * phiX * phiX
            except FloatingPointError:
                num = 0
            try:
                denom = (phiX * phiX) + (phiY * phiY) + epsilon
            except FloatingPointError:
                denom = 1

            if denom == 0.0:
                print('Adjusting for Zero Division')
                dtPhi[y, x] = 0
            else:
                dtPhi[y, x] = num/denom

    return dtPhi


def frontProp(phi, wX, wY):
    ''' Helper function to calculate fron Propagation term
    arguments:
    wX      -- gradient of image along x axis
    wY      -- gradient of image along y axis
    dtPhi   -- matrix of change in phi values
    returns:
    dtPhi   -- updated matrix of change in phi values
    '''
    dtPhi = np.zeros(phi.shape)
    for y in range(1, phi.shape[0]-1):
        for x in range(1, phi.shape[1]-1):
            # Handle border pixels
            xMinus1 = max(0, x-1)
            yMinus1 = max(0, y-1)
            xPlus1 = min(phi.shape[1]-1, x+1)
            yPlus1 = min(phi.shape[0]-1, y+1)
            try:
                # Slide 76 uphill gradient
                dtPhi[y, x] = max(wX[y, x], 0) * (phi[y, xPlus1] - phi[y, x]) + \
                    min(wX[y, x], 0) * (phi[y, x] - phi[y, xMinus1]) + \
                    max(wY[y, x], 0) * (phi[yPlus1, x] - phi[y, x]) + \
                    min(wY[y, x], 0) * (phi[y, x] - phi[yMinus1, x])
            except FloatingPointError:
                dtPhi[y, x] = 0

    return dtPhi


def drawSnake(ax1, ax2, phi, t):
    ''' Helper function to draw the Snake
    '''
    ax1.clear()
    ax1.imshow(Im, cmap='gray')
    ax1.set_title('frame ' + str(t))

    contour = get_contour(phi)
    if len(contour) > 0:
        ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

    ax2.clear()
    ax2.imshow(phi)
    # ax2.set_title(r'$\phi$', fontsize=22)
    ax2.set_title('phi', fontsize=22)
    print('Iteration no: {}'.format(t+1))
    plt.pause(0.01)


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here

    # ------------------------

    # Smooth the image and calculate gradients along X and Y directions
    smoothedIm = cv2.GaussianBlur(Im, (3, 3), 2*np.sqrt(2))
    wX = cv2.Sobel(smoothedIm, -1, 1, 0, ksize=3)
    wY = cv2.Sobel(smoothedIm, -1, 0, 1, ksize=3)

    # Magnitude of gradients
    w = np.sqrt(wX * wX + wY * wY)
    # Proposed metric
    w = 1/(1+w)

    # Approximate gradW along x and y directions - Slide 76
    for y in range(wY.shape[0]):
        wY[y, :] = 0.5 * (w[min(y+1, w.shape[0]-1), :] - w[max(0, y-1), :])
    for x in range(wX.shape[1]):
        wX[:, x] = 0.5 * (w[:, min(x+1, w.shape[1]-1)] - w[:, max(0, x-1)])

    # Define the constants
    tau = 1 / (4 * np.max(w))
    epsilon = 0.0001

    # Without this, the first plt show does not work
    drawSnake(ax1, ax2, phi, -1)

    startTime = time.time()
    for t in range(n_steps):

        # ------------------------
        # your implementation here

        # ------------------------

        dtPhi = tau * w * meanCurvature(phi, epsilon)
        dtPhi += frontProp(phi, wX, wY)
        phi += dtPhi

        if t % plot_every_n_step == 0:
            drawSnake(ax1, ax2, phi, t)

    plt.show()
    endTime = time.time()
    print('Total time taken: {} seconds'.format(endTime - startTime))
