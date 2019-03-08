import numpy as np
import utils

# ======================= PCA =======================
def pca(covariance, preservation_ratio=0.9):

    # Happy Coding! :)
    U, L2, Ut = np.linalg.svd(covariance)
    # Use cumsum to reduce the values of K
    pc_weights = np.cumsum(L2)/np.sum(L2)
    pc_weights = pc_weights[pc_weights < preservation_ratio]
    # Slide 55 and Chapter 17 (Prince)
    K = pc_weights.size
    sigma2 = np.sum(L2[K:L2.shape[0]])/(U.shape[0] - K)
    pcs = np.dot(U[:, 0:K], np.diag(np.sqrt(L2[0:K] - sigma2)))

    return pcs, pc_weights




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    # Slide 55 and Chapter 17 (Prince)
    W = kpts - mean_shape
    W = W.reshape(W.shape[0],-1)
    return np.dot(W, W.T)





# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    # your part here
    # Section 17.5.2 (Prince)
    for wtIdx in range(pc_weights.shape[0]):
        stackShapes = mean.copy()
        for i in range(1, 3):
            tempShape = mean.copy()
            tempShape = tempShape.reshape(-1,2)
            # Set hidden weights as -1 * pc_weights or 1 * pc_weights
            weights = pcs[:,wtIdx] * pow(-1, i) * pc_weights[wtIdx]
            tempShape[0:pcs.shape[0], :] += weights.reshape(-1,1)
            stackShapes = np.vstack((stackShapes, tempShape[np.newaxis]))
        utils.visualize_hands(stackShapes, 'Component Weight=' + str(pc_weights[wtIdx]))




# ======================= Training =======================
def train_statistical_shape_model(kpts):
    # Your code here    
    meanShape = np.mean(kpts, axis=0)
    meanShape = meanShape[np.newaxis, ...]
    coVar = create_covariance_matrix(kpts, meanShape)
    pcs, pc_weights = pca(coVar)
    visualize_impact_of_pcs(meanShape, pcs, pc_weights)

    return meanShape, pcs, pc_weights



# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):
    #ToDo
    ref = mean.copy()
    recon = kpts.copy()
    ref = np.tile(ref,(pc_weight.shape[0],1,1))
    hidWeights = (pcs * pc_weight).T
    print('------------------------------------')
    print('Hidden Weights:')    
    print(hidWeights)
    ref[:,0:pcs.shape[0],:] += hidWeights[..., np.newaxis]
    error = np.sqrt(np.sum(np.square(ref - mean), axis=2))/ref.shape[1]
    error = np.sum(error, axis=1)
    print('------------------------------------')
    print('RMS Errors: {}'.format(error))
    # Create the reconstruction for hidden weights with least error
    ref = ref[np.argmin(error)]
    ref = ref[np.newaxis, ...]
    recon = np.vstack((recon, ref))
    utils.visualize_hands(mean, 'Mean Shape')
    utils.visualize_hands(recon, 'Original+Reconstructed Shape')
    