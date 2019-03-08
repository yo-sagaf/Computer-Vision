import numpy as np
import utils



# ========================== Mean =============================
def calculate_mean_shape(kpts):
    # ToDO
    return np.mean(kpts, axis=0)



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    # Happy Coding
    # Slide 31-32
    for idx in range(kpts.shape[0]):        
        A = np.zeros((112,6))
        A[::2, 0:2] = reference_mean
        A[1::2, 2:4] = reference_mean
        A[::2, 4] = 1
        A[1::2, 5] = 1
        psedInvA = np.linalg.pinv(A)
        mean = np.zeros((reference_mean.size,1))
        mean[::2,0] = reference_mean[:,0]
        mean[1::2,0] = reference_mean[:,1]
        transVec = np.dot(psedInvA, mean)
        kpts[idx,:,0] = np.dot(A[::2], transVec).reshape(-1)
        kpts[idx,:,1] = np.dot(A[1::2],transVec).reshape(-1)
    return kpts



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):
    # ToDo
    mean = mean_shape.copy()
    error = np.sqrt(np.sum(np.square(kpts - mean), axis=2))/kpts.shape[1]
    error = np.sum(error, axis=1)
    return error




# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    aligned_kpts = kpts.copy()
    # utils.visualize_hands(kpts,'Before alignment')

    counter = 0
    oldError = np.zeros((kpts.shape[0]))
    for iter in range(max_iter):

        reference_mean = calculate_mean_shape(aligned_kpts)

        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        ##################### Your Part Here #####################

        ##########################################################
        error = compute_avg_error(kpts, reference_mean)
        if (oldError == error).all():
            counter += 1
        else:
            oldError = error
        if counter == 10:
            print('Shapes have converged')
            print('----------------------------------------')
            break
        print('Iteration = {} || error = {}'.format(iter, error))
        if np.min(error) <= 1e-5:
            break

    # visualize
    utils.visualize_hands(aligned_kpts, 'Aligned shapes')

    # visualize mean shape
    meanShape = calculate_mean_shape(aligned_kpts)
    meanShape = meanShape[np.newaxis, ...]
    utils.visualize_hands(meanShape, 'New Mean Shape')

    return aligned_kpts
