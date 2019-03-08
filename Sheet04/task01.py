import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------
def elastic_energy(current, previous, average_distance ):
    # The distance between the previous and the point being analysed
    euclideanDistance = np.sqrt( np.sum( ( previous - current ) ** 2 ) )
    # penalizing deviation from the average distance between pairs of nodes
    return (abs( euclideanDistance - average_distance ))**2

def external_energy( curr_point, sobelX, sobelY, Im ):
    height, width = Im.shape
    # to avoid out of bound error
    if curr_point[0] < 0 or curr_point[0] >= width or curr_point[1] < 0 or curr_point[1] >= height:
            return 999999
    return -( sobelX[ curr_point[1] ][ curr_point[0] ]**2 + sobelY[ curr_point[1] ][ curr_point[0] ]**2  )


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200

    # ------------------------
    # your implementation here

    # ------------------------
    #gradient along x direction
    sobelx = cv2.Sobel(Im,cv2.CV_64F,1,0,ksize=5)
    #gradient along y direction
    sobely = cv2.Sobel(Im,cv2.CV_64F,0,1,ksize=5)
    energy_matrix = np.zeros((len(V), 9), float) # search box size 3X3
    position_matrix = np.zeros((len(V), 9), int)
    neighbour = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1,0], [1,1]]
    #base case
    # for l in range(9):
    #     energy_matrix[0][l] = 0.0
    alpha = 0.8
    for t in range(n_steps):
        # ------------------------
        # your implementation here
        for i in range(1,len(V)):
            len_V = len(V)
            previous_point = V[(i + len_V - 1) % len_V]
            current_point = V[i]
            next_point  = V[(i+1) % len_V]
            # iterating over the neighbours 
            for j in range(9):
                min_energy = 999999
                for k in range(9):
                    euclidean_sum_V = 0.0
                    for q in range(len(V)):
                        # calculate the euclidean length of V
                        euclidean_sum_V += np.sqrt( np.sum( ( V[i] - V[ (i+1)%len(V) ] ) ** 2 ) )
                    avg_distance = euclidean_sum_V / len(V)
                    c_p = np.array( [current_point[0] + (j), current_point[1] + (k)] )
                    
                    # external energy
                    U = external_energy(c_p, sobelx, sobely, Im)
                    
                    # internal energy which is the pair wise cost
                    pairwise_cost = alpha * elastic_energy(c_p, previous_point, avg_distance)
                    
                    # total cost
                    energy = energy_matrix[i-1][k] +  U + pairwise_cost
                    # finding the minimum energy
                    if energy < min_energy:
                        min_energy = energy
                        pos = k
               
            #storing the minimum energy in dp table
            energy_matrix[i][j] = min_energy
            position_matrix[i][j] = pos
            
        
        min_energy_last_column =  99999999
        for m in range(9):
            if energy_matrix[19][m] < min_energy_last_column:
                # retrieving the minimal possible cost to reach last column
                min_energy_last_column = energy_matrix[19][m]
                #index of minimum cost in the last column
                position_last_column =  m

        pos = position_last_column

        # backtracking to find optimal route
        for i in range(len(V)-1, 0, -1 ):
            # updating snake coordinates
            print (V[i])
            V[i] = np.add( V[i], neighbour[pos])
            pos = position_matrix[i-1][pos]

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
