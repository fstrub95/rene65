import numpy as np
import cv2
from sample import Sample
from matplotlib import pyplot as plt, cm

from misc.logger import create_logger
import random

class ScoreToLow(Exception): pass


def show_img(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img)
    cv2.waitKey(0)

min_row = 0
max_row = 1565
min_col = 700
max_col = 2715

logger = create_logger("../data/grain/log.txt")

segment = Sample("../data/segment/Slice70.tif")
segment = segment.get_image()
segment = segment[min_row:max_row, min_col:max_col]
segment = cv2.resize(segment, None, fx=0.3265, fy=0.3265, interpolation = cv2.INTER_AREA)
segment = cv2.bitwise_not(segment)


grain = Sample("../data/grain/Slice70wDistorsions.bmp")
grain = grain.get_image()
grain = cv2.bitwise_not(grain)


(width, height) = grain.shape
max_score = 0
best_translation, best_angle = None, None
best_grain, best_segment = None, None

# Best score sans distorsion 1577 1.75 (-10, 122)


for _ in range(1,100000):

    distCoeff = np.zeros((4, 1), np.float64)

    # TODO: add your coefficients here!
    k1 = random.choice([-1, 1])*10**(-random.uniform(2,7))  # negative to remove barrel distortion
    k2 = random.choice([-1, 1])*10**(-random.uniform(2,8))  # negative to remove barrel distortion
    p1 = random.choice([-1, 1])*10**(-random.uniform(2,7))
    p2 = random.choice([-1, 1])*10**(-random.uniform(2,8))

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    focal = random.uniform(1, 20)
    cam[0, 2] = random.gauss(width / 2.0, 4)  # define center x
    cam[1, 2] = random.gauss(height / 2.0, 4)  # define center y
    cam[0, 0] = focal  # define focal length x
    cam[1, 1] = focal  # define focal length y

    # here the undistortion will be computed
    grain_distord = cv2.undistort(grain, cam, distCoeff)

    try:
        for angle in np.arange(1, 3.0, 0.25):

            M = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
            rot_grain = cv2.warpAffine(grain_distord, M, (height, width))

            rot_grain[rot_grain < 128] = 0
            rot_grain[rot_grain >= 128] = 255

            for tx in range(-25, 5, 1):
                for ty in range(105, 135, 1):

                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    segment_translate = cv2.warpAffine(segment, M, (height, width))

                    rot_grain[rot_grain < 128] = 1
                    rot_grain[rot_grain >= 128] = 255  # put a different background

                    score = (rot_grain == segment_translate).sum()

                    if score < 450:
                        print("skip", score)
                        raise ScoreToLow

                    if score > max_score:
                        max_score = score
                        best_grain, best_segment = rot_grain, segment_translate
                        best_val = (max_score, (tx, ty), angle, [k1, k2, p1, p2], cam)

                        logger.info(("---> ", best_val))

                        fig = plt.figure(figsize=(15, 8))
                        plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
                        plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
                        fig.savefig("out.png")
    except ScoreToLow:
        continue

logger.info("")
logger.info("BEST SCORE:")
logger.info(("###> ", best_val))

fig = plt.figure(figsize=(15, 8))
plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
fig.savefig("out.png")

# link to fix distorsion:
# https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image






