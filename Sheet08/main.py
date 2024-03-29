import numpy as np
import time

import utils
import task1
import task2

hands_orig_train = 'data/hands_orig_train.txt.new'
hands_aligned_test = 'data/hands_aligned_test.txt.new'
hands_aligned_train = 'data/hands_aligned_train.txt.new'

def get_keypoints(path):
    data_info = utils.load_data(path)

    # Your part here
    kpts = utils.convert_samples_to_xy(data_info['samples']) 

    return kpts

def task_1():
    # Loading Trainig Data
    kpts = get_keypoints(hands_orig_train)
    
    # calculate mean
    # ToDO
    meanShape = task1.calculate_mean_shape(kpts)
    meanShape = meanShape[np.newaxis, ...]

    # we want to visualize the data first
    # ToDO
    utils.visualize_hands(meanShape, 'Mean Shape')

    task1.procrustres_analysis(kpts)
    time.sleep(5)


def task_2_1():
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_train)

    ### Your part here ##

    #####################

    mean, pcs, pc_weights = task2.train_statistical_shape_model(kpts)

    return mean, pcs, pc_weights

def task_2_2(mean, pcs, pc_weights):
    # ============= Load Data =================

    # Your part here

    kpts = get_keypoints(hands_aligned_test)

    task2.reconstruct_test_shape(kpts, mean, pcs, pc_weights)

    time.sleep(5)

if __name__ == '__main__':
    print("Running Task 1")
    task_1()

    print("Running Task 2.1")
    mean, pcs, pc_weights = task_2_1()

    print("Running Task 2.2")
    task_2_2(mean, pcs, pc_weights)
