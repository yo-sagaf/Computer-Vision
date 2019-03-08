import cv2
import numpy as np
import scipy.stats as st
import time
import math


def displayImage(winName, img):
    """ Helper function to display image
    arguments:
    winName -- Name of display window
    img     -- Source Image
    """
    cv2.imshow(winName, img)
    cv2.waitKey(0)


def get_convolution_using_fourier_transform(image, kernel):
    """ Helper function to filter image using Fourier Transform
    arguments:
    image  -- Source Image
    kernel -- Input Kernel/Filter
    """
    imageFFT = np.fft.fft2(image)
    imageFFT = np.fft.fftshift(imageFFT)
    kernelFFT = np.fft.fft2(kernel, image.shape)
    kernelFFT = np.fft.fftshift(kernelFFT)
    blurImg = np.fft.ifft2(imageFFT * kernelFFT)
    return np.absolute(blurImg)


def task1():
    print('Task 1')
    print("------------------------------------------------------------")

    image = cv2.imread('../data/einstein.jpeg', 0).astype('float64')
    image /= 255
    displayImage('Image', image)

    # get 1-D kernel
    kernel = cv2.getGaussianKernel(7, 1)
    # convert to 2-D kernel
    kernel = np.dot(kernel, np.transpose(kernel))

    # calculate convolution of image and kernel
    conv_result = cv2.filter2D(image, -1, kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    displayImage('Convolution blur', conv_result)
    displayImage('FFT blur', np.absolute(fft_result))

    # compare results
    print('Mean pixel-wise difference = ' +
          str(np.abs(conv_result - fft_result).mean()))

    print("============================================================")


def normalized_cross_correlation(image, template):
    """ Helper function to match templates using NCR method
    arguments:
    image    -- Source Image
    template -- Source Template
    """
    h = np.zeros((image.shape[0] - template.shape[0] + 1,
                  image.shape[1] - template.shape[1] + 1), dtype=float)
    tempCor = (template - np.mean(template)).astype('float64')
    for y in range(h.shape[0]):
        for x in range(h.shape[1]):
            imagePatch = (image[y:y+template.shape[0], x:x +
                                template.shape[1]]).astype('float64')
            imagePatch -= np.mean(imagePatch)
            num = np.sum(tempCor * imagePatch)
            denom = np.sqrt(np.sum(np.square(tempCor)) *
                            np.sum(np.square(imagePatch)))
            h[y, x] = num/denom
    return h


def task2():
    print('Task 2')
    print("------------------------------------------------------------")

    image = cv2.imread('../data/lena.png', 0)
    template = cv2.imread('../data/eye.png', 0)
    displayImage('Original Image', image)
    displayImage('Template', template)

    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)

    result_ncc = normalized_cross_correlation(image, template)

    # draw rectangle around found location in all four results
    # show the results
    imageCustom = image.copy()

    _, _, _, max_loc = cv2.minMaxLoc(res)
    # The opencv mapping matrix seems weird compared to the custom one. Unable to use the same logic
    # as that used in Customer to filter bounding boxes
    cv2.rectangle(
        image, max_loc, (max_loc[0] + template.shape[0], max_loc[1] + template.shape[1]), 255, 2)
    displayImage('OpenCV Match Template', image)

    boundingBoxCustom = np.where(result_ncc >= 0.7)
    for i in range(len(boundingBoxCustom[0])):
        topLeft = (boundingBoxCustom[0][i], boundingBoxCustom[1][i])
        bottomRight = (boundingBoxCustom[0][i] + template.shape[0],
                       boundingBoxCustom[1][i] + template.shape[1])
        cv2.rectangle(imageCustom, topLeft, bottomRight, 255, 2)
    displayImage('Custom Match Template', imageCustom)

    print("============================================================")


def build_gaussian_pyramid_opencv(image, num_levels):
    """ Helper function to build Gaussian Pyramid using OpenCV
    arguments:
    image      -- Source Image
    num_levels -- Number of Levels of Pyramid
    """
    GP = image.copy()
    gpA = [GP]
    for i in range(num_levels):
        GP = cv2.pyrDown(GP)
        gpA.append(GP)
    gpA.reverse()
    return gpA


def build_gaussian_pyramid(image, num_levels, sigma):
    """ Helper function to build Custom Gaussian Pyramid
    arguments:
    image      -- Source Image
    num_levels -- Number of Levels of Pyramid
    sigma      -- S.D. for gaussian kernel
    """
    image_copy = image.copy()
    gpA = [image_copy]
    kernel_size = 3
    for i in range(num_levels):
        image_blur = cv2.GaussianBlur(
            image_copy, (kernel_size, kernel_size), sigma)
        img_Gaussian_Blur = image_blur[::2, ::2]
        # img_Gaussian_Blur = cv2.resize(
        # image_blur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        kernel_size = kernel_size * 2 - 1
        image_copy = img_Gaussian_Blur
        gpA.append(img_Gaussian_Blur)
    gpA.reverse()
    return gpA


def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
    """ Helper function to match template using Gaussian Pyramid
    arguments:
    pyramid_image    -- Gaussian Pyramid of Source Image
    pyramid_template -- Gaussian Pyramid of Template
    threshold        -- Threshold for matching
    """
    start = time.time()
    results = []
    thrshld = None

    # Do template match
    for idx in range(5):
        refimg = pyramid_image[idx]
        tplimg = pyramid_template[idx]

        # On the first level performs regular template matching.
        if idx == 0:
            result = cv2.matchTemplate(refimg, tplimg, cv2.TM_CCORR_NORMED)
        # On other levels, perform pyramid transformation and template matching
        # on the predefined ROI areas, obtained by result of the previous level.
        else:
            mask = cv2.pyrUp(thrshld)
            mask8u = cv2.inRange(mask, 0, 255)
            # contours to define the region of interest and perform template matching on the areas.
            _, contours, _ = cv2.findContours(
                mask8u, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)

            tH, tW = tplimg.shape[:2]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                src = refimg[y:y+h+tH, x:x+w+tW]
                result = cv2.matchTemplate(src, tplimg, cv2.TM_CCORR_NORMED)

        _, thrshld = cv2.threshold(result, threshold, 1., cv2.THRESH_TOZERO)
        results.append(thrshld)

    # Analyse the result
    result = thrshld
    refimg = pyramid_image[len(pyramid_image)-1]
    tplimg = pyramid_template[len(pyramid_template)-1]
    _, maxval, _, maxloc = cv2.minMaxLoc(result)
    if maxval > threshold:
        pt1 = maxloc
        pt2 = (maxloc[0] + tplimg.shape[1], maxloc[1] + tplimg.shape[0])
        print("Found the template region using OpenCV: {} => {}".format(pt1, pt2))
        cv2.rectangle(refimg, pt1, pt2, 255, 2)
        displayImage("OpenCV Template Match", refimg)
    else:
        print("Cannot find the template in the origin image!")

    end = time.time()
    return (end-start)


def task3():
    print('Task 3')
    print("------------------------------------------------------------")

    image = cv2.imread('../data/traffic.jpg', 0)
    template = cv2.imread('../data/traffic-template.png', 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
    mine_pyramid = build_gaussian_pyramid(image, 4, 1)

    # compare and print mean absolute difference at each level
    for i in range(4):
        # displayImage('CV Level' + str(i), cv_pyramid[i])
        # displayImage('Mine Level' + str(i), mine_pyramid[i])
        # Mean difference is pretty high, possibly because of difference in smoothing and resizing from OpenCV method
        # Could also be because of border handling
        print('Mean absolute difference at Level ' + str(i) + ' = ' +
              str(np.abs(cv_pyramid[i] - mine_pyramid[i]).mean()))
        print("------------------------------------------------------------")

    # fast template matching
    pyramid_template = build_gaussian_pyramid(template, 4, 2.5)
    result = template_matching_multiple_scales(
        mine_pyramid, pyramid_template, 0.7)

    # show result
    print('Time taken by fast template matching: ' + str(result))
    print("------------------------------------------------------------")

    # performance calculation of normalised cross correlation
    start = time.time()
    result_ncc = normalized_cross_correlation(image, template)
    end = time.time()
    print('Time taken by normalised cross correlation: ' + str(end - start))
    print("------------------------------------------------------------")

    _, _, _, maxloc = cv2.minMaxLoc(result_ncc)
    topLeft = maxloc
    bottomRight = (maxloc[0] + template.shape[1],
                   maxloc[1] + template.shape[0])
    print("Found the template region using NCC: {} => {}".format(
        topLeft, bottomRight))
    cv2.rectangle(image, topLeft, bottomRight, 255, 2)
    displayImage("NCC Template Match", image)

    print("============================================================")


def get_derivative_of_gaussian_kernel(size, sigma):
    """ Helper function to find derivative of a kernel
    arguments:
    size  -- Kernel Size
    sigma -- S.D. of the gaussian kernel
    """
    # 1d kernel
    gaussian_kernel = cv2.getGaussianKernel(size, sigma)
    # 2d kernel
    gaussian_kernel = np.dot(gaussian_kernel, np.transpose(gaussian_kernel))
    # derivatives
    dy, dx = np.gradient(gaussian_kernel)

    return dx, dy


def task4():
    print('Task 4')
    print("------------------------------------------------------------")

    image = cv2.imread('../data/einstein.jpeg', 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)
    edges_x = cv2.filter2D(image, -1, kernel_x)  # convolve with kernel_x
    edges_y = cv2.filter2D(image, -1, kernel_y)  # convolve with kernel_y

    # edges in x and y directions
    displayImage('Edges along x-axis', edges_x)
    displayImage('Edges along y-axis', edges_y)

    magnitude = cv2.magnitude(edges_x.astype(
        np.float), edges_y.astype(np.float))
    direction = np.arctan2(edges_y, edges_x)  # compute edge direction

    displayImage('Magnitude', magnitude.astype(np.float))
    displayImage('Direction', direction.astype(np.float))

    print("============================================================")


###################################################################################################################
# Inspired from the code by https://stackoverflow.com/questions/36272985/generalized-distance-transform-in-python #
# Unfortunately does not work presently. Only returns a completely black Image                                    #
###################################################################################################################
def distance_transform_1d(srcVec, inSize, positive_inf, negative_inf):
    """ Distance transform of 1d function using squared distance 
    arguments:
    srcVec -- Source 1D Array
    inSize -- Size of Array
    positive_inf -- Positive Infinity
    negative_inf -- Negative Infinity
    """
    distTransVec = np.zeros(srcVec.shape)
    k = 0
    v = np.zeros(inSize, dtype=int)
    z = np.zeros(inSize + 1)
    v[0] = 0
    z[0] = negative_inf
    z[1] = positive_inf

    # Algorithm 1 from Pedro Felzenszwalb and Daniel Huttenlocher. "Distance transforms of sampled functions"
    for q in range(1, inSize):
        s = (((srcVec[q] + q * q) - (srcVec[v[k]] +
                                     v[k] * v[k])) / (2.0 * q - 2.0 * v[k]))
        while s <= z[k]:
            k -= 1
            s = (((srcVec[q] + q * q) - (srcVec[v[k]] +
                                         v[k] * v[k])) / (2.0 * q - 2.0 * v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = positive_inf

    k = 0
    for q in range(inSize):
        while z[k + 1] < q:
            k += 1
        distTransVec[q] = ((q - v[k]) * (q - v[k]) + srcVec[v[k]])
    distTransVec = np.clip(distTransVec, 0, 255)
    return distTransVec


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    """ Distance transform of 2d function using squared distance 
    arguments:
    edge_function -- Source 2D Array
    positive_inf  -- Positive Infinity
    negative_inf  -- Negative Infinity
    """
    rows, cols = edge_function.shape
    f = np.zeros(max(rows, cols))
    # transform along columns
    for x in range(cols):
        f = edge_function[:, x]
        edge_function[:, x] = distance_transform_1d(
            f, rows, positive_inf, negative_inf)
    # transform along rows
    for y in range(rows):
        f = edge_function[y, :]
        edge_function[y, :] = distance_transform_1d(
            f, cols, positive_inf, negative_inf)
    return edge_function


def task5():
    print('Task 5')
    print("------------------------------------------------------------")

    image = cv2.imread('../data/traffic.jpg', 0)
    displayImage('Traffic', image)

    image = cv2.GaussianBlur(image, (3, 3), 2, borderType=cv2.BORDER_DEFAULT)

    edges = cv2.Canny(image, 230, 350)  # compute edges
    displayImage('Gradient Image', edges)

    _, edges = cv2.threshold(
        edges, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    edge_function = edges.copy()  # prepare edges for distance transform

    start_time = time.time()
    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, np.Inf, np.NINF)
    end_time = time.time()
    print('Time taken by custom Distance Transform Function: ' +
          str(round(end_time - start_time, 5)) + ' seconds')
    print("------------------------------------------------------------")

    displayImage('Custom Distance Transform', dist_transfom_mine)

    # Pedro Felzenszwalb and Daniel Huttenlocher. "Distance transforms of sampled functions" as per OpenCV tutorials
    dist_transfom_cv = cv2.distanceTransform(
        edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # compute using opencv
    cv2.normalize(dist_transfom_cv, dist_transfom_cv, 0, 1.0, cv2.NORM_MINMAX)
    displayImage('Distance Transform Image', dist_transfom_cv)

    # compare and print mean absolute difference
    print('Mean pixel-wise difference = ' +
          str(np.abs(dist_transfom_mine - dist_transfom_cv).mean()))

    print("============================================================")


if __name__ == '__main__':
    task1()
    cv2.destroyAllWindows()
    task2()
    cv2.destroyAllWindows()
    task3()
    cv2.destroyAllWindows()
    task4()
    cv2.destroyAllWindows()
    task5()
    cv2.destroyAllWindows()
