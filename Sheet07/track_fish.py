import cv2
import numpy as np
import os

IMAGES_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'images')

INIT_X = 448
INIT_Y = 191
INIT_WIDTH = 38
INIT_HEIGHT = 33

INIT_BBOX = [INIT_X, INIT_Y, INIT_WIDTH, INIT_HEIGHT]


def load_frame(frame_number):
    """
    :param frame_number: which frame number, [1, 32]
    :return: the image
    """
    image = cv2.imread(os.path.join(IMAGES_FOLDER, '%02d.png' % frame_number))
    return image


def crop_image(image, bbox):
    """
    crops an image to the bounding box
    """
    x, y, w, h = tuple(bbox)
    return image[y: y + h, x: x + w]


def draw_bbox(image, bbox, thickness=2, no_copy=False):
    """
    (optionally) makes a copy of the image and draws the bbox as a black rectangle.
    """
    x, y, w, h = tuple(bbox)
    if not no_copy:
        image = image.copy()
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness)

    return image


def compute_histogram(image):
    # implement here
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


def compare_histogram(hist1, hist2):
    # implement here

    hist_comp_val = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    likelihood = np.exp(-hist_comp_val * 20.0)
    return likelihood


class Position(object):
    """
    A general class to represent position of tracked object.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bbox(self):
        """
        since the width and height are fixed, we can do such a thing.
        """
        return [self.x, self.y, INIT_WIDTH, INIT_HEIGHT]

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Position(self.x * other, self.y * other)

    def __repr__(self):
        return "[%d %d]" % (self.x, self.y)

    def make_ready(self, image_width, image_height):
        # convert to int
        self.x = int(round(self.x))
        self.y = int(round(self.y))

        # make sure inside the frame
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.x = min(self.x, image_width)
        self.y = min(self.y, image_height)


class ParticleFilter(object):
    def __init__(self, du, sigma, num_particles=200):
        # the template (histogram) of the object that is being tracked.
        self.template = None
        self.position = None  # we don't know the initial position still!
        # we will store list of particles at each step here for displaying.
        self.particles = []
        self.fitness = []  # particle's fitness values
        self.du = du
        self.sigma = sigma
        self.num_particles = num_particles
        self.mean = None
        self.tempHist = None

    def init(self, frame, bbox):
        # initializing the position
        self.position = Position(x=bbox[0], y=bbox[1])
        # implement here ...
        self.mean = 0
        topLeftX, topLeftY, boxWidth, boxHeight = self.position.get_bbox()
        self.tempHist = compute_histogram(frame[topLeftY:topLeftY+min(
            boxHeight, frame.shape[0]), topLeftX:topLeftX+min(boxWidth, frame.shape[1])])
        for idx in range(self.num_particles):
            self.particles.append(
                Position(x=self.position.x, y=self.position.y))
            topLeftX, topLeftY, boxWidth, boxHeight = self.particles[idx].get_bbox(
            )
            self.fitness.append(0.0)

    def track(self, new_frame):
        # implement here ...
        for i in range(self.num_particles):
            # Step 1 Compute Weights
            topLeftX, topLeftY, boxWidth, boxHeight = self.particles[i].get_bbox()
            parHist = compute_histogram(new_frame[topLeftY:topLeftY+min(
                boxHeight, new_frame.shape[0]), topLeftX:topLeftX+min(boxWidth, new_frame.shape[1])])
            self.fitness[i] = compare_histogram(self.tempHist, parHist)

            # Resample
            j = self.resample()
            
            # Motion model application seems wrong. Not sure if we are to update the fitness at each step
            # or how to update the mean

            # Step 2 Add Noise
            # Without multiplying du, causes invalid value error
            x = self.particles[j].x
            y = self.particles[j].y
            self.particles[j] = Position(x=x+self.du*np.random.normal(
                self.mean, self.sigma**2), y=y+self.du*np.random.normal(self.mean, self.sigma**2))
            self.particles[j].make_ready(
                new_frame.shape[1], new_frame.shape[0])

            # Update Mean
            if j+1 < self.num_particles:
                self.mean = self.du*self.mean + (1-self.du)*(self.fitness[j+1]-self.fitness[j])

    def resample(self):
        randNo = np.random.uniform(0, 1)
        j = 0
        while (j < self.num_particles-1 and randNo > self.fitness[j]):
            j += 1
        return j

    def display(self, current_frame):
        cv2.imshow('frame', current_frame)

        frame_copy = current_frame.copy()
        for i in range(len(self.particles)):
            draw_bbox(frame_copy, self.particles[i].get_bbox(
            ), thickness=1, no_copy=True)

        cv2.imshow('particles', frame_copy)
        cv2.waitKey(100)


def main():
    np.random.seed(0)
    DU = 0.5
    SIGMA = 15

    cv2.namedWindow('particles')
    cv2.namedWindow('frame')
    frame_number = 1
    frame = load_frame(frame_number)

    tracker = ParticleFilter(du=DU, sigma=SIGMA)
    tracker.init(frame, INIT_BBOX)
    tracker.display(frame)

    for frame_number in range(2, 33):
        frame = load_frame(frame_number)
        tracker.track(frame)
        tracker.display(frame)


if __name__ == "__main__":
    main()
