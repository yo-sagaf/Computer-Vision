import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, Lambda, sigma_p, Phi, sigma_m):
        self.Lambda = Lambda
        self.sigma_p = sigma_p
        self.Phi = Phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.mean_p = None
        self.mean_m = None

    def init(self, init_state):
        self.state = init_state
        self.convariance = np.eye(init_state.shape[0]) * 0.01
        self.mean_p = np.array([0,0,0,0]).T
        self.mean_m = np.array([0,0]).T

    def track(self, xt):
        # implement here

        mean_plus = self.mean_p + self.Lambda @ self.state
        sigma_plus = self.sigma_p + (self.Lambda @ self.convariance) @ self.Lambda.T
        inverse_term = np.linalg.inv(self.sigma_m + ((self.Phi @ sigma_plus) @ self.Phi.T))
        kalman_gain = (sigma_plus @ self.Phi.T) @ inverse_term
        ## State Update
        mean_t = mean_plus + kalman_gain @ (xt - self.mean_m - (self.Phi @ mean_plus))
        identity_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ## Covariance Update
        sigma_t = (identity_matrix - (kalman_gain @ self.Phi)) @ sigma_plus

        self.state = mean_t
        self.convariance = sigma_t




    def get_current_location(self):
        return self.Phi @ self.state


def main():
    init_state = np.array([0, 1, 0, 0])

    Lambda = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    Phi = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    sm = 0.05
    sigma_m = np.array([[sm, 0], [0,sm]])

    tracker = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker.init(init_state)

    track = []
    for t in range(len(observations)):
        print(get_observation(t))
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.show()


if __name__ == "__main__":
    main()
