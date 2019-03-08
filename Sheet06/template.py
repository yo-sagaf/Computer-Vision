#!/usr/bin/python3.5

import numpy as np
from scipy import misc


'''
    read the usps digit data
    returns a python dict with entries for each digit (0, ..., 9)
    dict[digit] contains a list of 256-dimensional feature vectores (i.e. the gray scale values of the 16x16 digit image)
'''


def read_usps(filename):
    data = dict()
    with open(filename, 'r') as f:
        N = int(np.fromfile(f, dtype=np.uint32, count=1, sep=' '))
        for n in range(N):
            c = int(np.fromfile(f, dtype=np.uint32, count=1, sep=' '))
            tmp = np.fromfile(f, dtype=np.float64, count=256, sep=' ') / 1000.0
            data[c] = data.get(c, []) + [tmp]
    for c in range(len(data)):
        data[c] = np.stack(data[c])
    return data


'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''


def read_face_image(filename):
    image = misc.imread(filename) / 255.0
    bounding_box = np.zeros(image.shape)
    bounding_box[50:100, 60:120, :] = 1
    foreground = image[bounding_box == 1].reshape((50 * 60, 3))
    background = image[bounding_box == 0].reshape((40000 - 50 * 60, 3))
    return image, foreground, background


'''
    implement your GMM and EM algorithm here
'''


class GMM(object):
    def __init__(self):
        self.mean = None
        self.covar = None
        self.lambdaK = None
        self.resp = None

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''

    def fit_single_gaussian(self, data):
        # TODO
        self.mean = np.mean(data, axis=0).reshape(-1,1)
        self.covar = np.diag(np.diag(np.cov(data, rowvar=False)))
        self.lambdaK = np.ones((1, 1))

    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations

        Comments - The log likelihood to make the calculations numerically stable is not working.
        We are not sure how to take care of the invalid data as in how to to set the lower limit.
        So there is a warning while calculating the covar matrix sometimes.
    '''

    def em_algorithm(self, data, n_iterations=10):
        # TODO
        noFeatures = data.shape[1]
        for iter in range(n_iterations):
            noMixtures = self.lambdaK.shape[1]
            self.resp = np.zeros((noFeatures, noMixtures))
            # Expection Step
            for k in range(noMixtures):
            # Log likelihood is not working here
                self.resp[:, k] = self.lambdaK[:, k] * np.random.multivariate_normal(
                    self.mean[:, k], self.covar[:, k*noFeatures:k*noFeatures+noFeatures])
            self.resp /= self.resp.sum(axis=0)

            # Maximization Step
            for k in range(noMixtures):
                self.lambdaK[:, k] = np.sum(self.resp[:, k])/np.sum(self.resp)
                self.mean[:, k] = np.sum(
                    self.resp[:, k] * data, axis=0) / np.sum(self.resp[:, k])
                tempCovar = np.diag(
                    self.covar[:, k*noFeatures:k*noFeatures+noFeatures])
                self.covar[:, k*noFeatures:k*noFeatures+noFeatures] = np.diag(self.resp[:, k] * tempCovar) / np.sum(self.resp[:, k])

    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
    def split(self, epsilon=0.1):
        # TODO
        self.lambdaK = np.hstack((self.lambdaK, self.lambdaK))/2.0
        newMean = np.zeros((self.mean.shape[0], self.mean.shape[1]*2))
        for i in range(self.mean.shape[1]):
            newMean[:, 2*i] = self.mean[:, i] + epsilon * np.diag(self.covar)
            newMean[:, 2*i+1] = self.mean[:, i] - epsilon * np.diag(self.covar)
        self.mean = newMean
        self.covar = np.hstack((self.covar, self.covar))

    '''
        sample a D-dimensional feature vector from the GMM
    '''
    def sample(self):
        # TODO
        noFeatures = self.mean.shape[0]
        noMixtures = self.lambdaK.shape[1]
        vec = np.zeros((noFeatures, noMixtures))
        for k in range(noMixtures):
            vec[:, k] = self.lambdaK[:, k] * np.random.multivariate_normal(
                    self.mean[:, k], self.covar[:, k*noFeatures:k*noFeatures+noFeatures])

        if noMixtures > 1:
            vec = np.sum(vec, axis=1)
        return vec

'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here

    Comments - The GMM for USPS does not perform very well for 100 iterations of each digit. 
    So we run it for the default 10 times.
'''
print('Training in progress for USPS Dataset.....')
data = read_usps('usps.txt')
gmm = [GMM() for _ in range(10)]  # 10 GMMs (one for each digit)
for split in [0, 1, 2]:
    result_image = np.zeros((160, 160))
    for digit in range(10):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit], n_iterations=10)
        # sample 10 images for this digit
        for i in range(10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon=0.1)

print('Samples generated for USPS Dataset')
print('-----------------------------------')

'''
    Task 2e: skin color model
    
    Comments - The code for 2e is only done to train the gmms on foreground and background. 
    We did not understand how to plug in the actual image data to the trained gmm 
    without reinitializing the weights, mean and covar matrices. So it is incomplete.
'''
image, foreground, background = read_face_image('face.jpg')
# print('Training in progress for Image foreground and background.....')
'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such 
    that everything below the threshold is black, display the resulting image
'''
gmm = [GMM() for _ in range(2)]  # 2 GMMs (one for each of foreground and background)
for split in [0, 1, 2]:
    # train the model
    if split == 0:
        gmm[0].fit_single_gaussian(foreground)
        gmm[1].fit_single_gaussian(background)
    else:
        gmm[0].em_algorithm(foreground)
        gmm[1].em_algorithm(background)
    # split the components to have twice as many in the next run
    gmm[0].split(epsilon=0.1)
    gmm[1].split(epsilon=0.1)
# How to apply the trained gmms into the image?
# print('Sample generated for Image')
# print('-----------------------------------')