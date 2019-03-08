import cv2
import numpy as np
import maxflow

def question_3(I,rho=0.6,pairwise_cost_same=0.01,pairwise_cost_diff=0.5):

    img = I.copy().astype('float64')/255.0

    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes
    nodeids = g.add_grid_nodes(img.shape)

    ### 3) Compute Unary cost
    uCost = -np.log(np.power(rho, img) * np.power(1-rho, img))

    ### 4) Add terminal edges
    g.add_grid_tedges(nodeids, uCost, 1-uCost)

    ### 5) Add Node edges
    ### Vertical Edges
    verWeights = np.zeros(img.shape)
    for y in range(img.shape[0]-1):
        verWeights[y,:] = np.abs(img[y, :] - img[y+1, :])
    verWeights[verWeights > 0] = pairwise_cost_diff
    verWeights[verWeights == 0] = pairwise_cost_same

    structure = np.array(([1], [0], [1]))
    g.add_grid_edges(nodeids, weights=verWeights, structure=structure, symmetric=True)

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
    horWeights = np.zeros(img.shape)
    for x in range(img.shape[1]-1):
        horWeights[:,x] = np.abs(img[:, x] - img[:, x+1])
    horWeights[horWeights > 0] = pairwise_cost_diff
    horWeights[horWeights == 0] = pairwise_cost_same

    structure = np.array(([1, 0, 1]))
    g.add_grid_edges(nodeids, weights=horWeights, structure=structure, symmetric=True)

    ### 6) Maxflow
    g.maxflow()

    sgm = g.get_grid_segments(nodeids)
    Denoised_I = np.float_(np.logical_not(sgm))

    print('Mean absolute difference for rho = {} is {}'.format(rho, np.mean(np.abs(img-Denoised_I))))
    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), \
    cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()
    #print(labels)
    Denoised_I = np.zeros_like(I)
    img = I.copy().astype('float64') / 255.0


    ### Use Alpha expansion binary image for each label
    for i in range(len(labels)):

                    ### 1) Define Graph
                    g = maxflow.Graph[float]()

                    ### 2) Add pixels as nodes
                    nodeids = g.add_grid_nodes(img.shape)



                    ### 3) Compute Unary cost
                    uCost = np.ones(img.shape) * (1 - rho) / 2
                    uCost[I == labels[i]] = rho

                    ### 4) Add terminal edges
                    g.add_grid_tedges(nodeids, uCost, 1 - uCost)

                    alpha = labels[i]
                    beta = labels[(i + 1) % len(labels)]
                    gamma = labels[(i + 2) % len(labels)]

                    ### 5) Add Node edges
                    ### Vertical Edges
                    for y in range(I.shape[0] - 1):
                        for x in range(I.shape[1]):
                            if I[y, x] == alpha and I[y+1, x] == alpha:
                                uCost[y, x] = rho
                            elif I[y, x] == alpha and I[y+1, x] == beta:
                                g.add_edge(nodeids[y,x], nodeids[y+1, x], 1, 1)
                            elif I[y, x] == beta and I[y+1, x] == beta:
                                g.add_edge(nodeids[y, x], nodeids[y+1, x], 1, 1)
                            elif I[y, x] == beta and I[y+1, x] == gamma:
                                extranode = g.add_nodes(1)
                                g.add_edge(nodeids[y, x], extranode[0], 1, 10000)
                                g.add_edge(nodeids[y+1, x], extranode[0], 1, 10000)
                                g.add_tedge(extranode[0], 0, 10000)

                    ### Horizontal edges
                    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
                    for y in range(I.shape[0]):
                        for x in range(I.shape[1] - 1):
                            if I[y, x] == alpha and I[y, x + 1] == alpha:
                                uCost[y, x] = rho
                            elif I[y, x] == alpha and I[y, x+1] == beta:
                                g.add_edge(nodeids[y,x], nodeids[y,x+1],1,1)
                            elif I[y, x] == beta and I[y, x+1] == beta:
                                g.add_edge(nodeids[y, x], nodeids[y, x + 1],1,1)
                            elif I[y, x] == beta and I[y, x+1] == gamma:
                                extranode = g.add_nodes(1)
                                g.add_edge(nodeids[y, x], extranode[0], 1, 10000)
                                g.add_edge(nodeids[y, x+1], extranode[0], 1, 10000)
                                g.add_tedge(extranode[0], 0, 10000)

                    ### 6) Maxflow
                    g.maxflow()


                    sgm = g.get_grid_segments(nodeids)
                    binaryImage = np.float_(np.logical_not(sgm))
                    Denoised_I[binaryImage == 1] = alpha

    print('Mean absolute Difference for rho = {} is {}'.format(rho, np.mean(np.abs(I-Denoised_I))))
    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    
    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.15)
    print('-----------------------------------------------------------------------')
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.3)
    print('-----------------------------------------------------------------------')
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.6)
    print('-----------------------------------------------------------------------')

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



