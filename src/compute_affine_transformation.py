import numpy as np
import cv2
from sample import Sample
from matplotlib import pyplot as plt, cm

from misc.logger import create_logger
import subprocess

import cma




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
min_score = 0
best_translation, best_angle = None, None




M = cv2.getRotationMatrix2D((height / 2, width / 2), 1.5, 1)
rot_grain = cv2.warpAffine(grain, M, (height, width))
rot_grain[rot_grain < 128] = 1
rot_grain[rot_grain >= 128] = 255
cv2.imwrite("../data/tmp/grain.png", rot_grain)


M = np.float32([[1, 0, -10], [0, 1, 121]])


segment_translate = cv2.warpAffine(segment, M, (height, width))
segment_translate[segment_translate < 128] = 0
segment_translate[segment_translate >= 128] = 255
cv2.imwrite("../data/tmp/segment.png", segment_translate)


best_segment = segment_translate
best_grain = rot_grain



path = '../data/tmp/'

# Create meshgrid

x = np.array(range(0, 601, 100))
y = np.array(range(0, 601, 100))
xv, yv = np.meshgrid(x, y)
xv = xv.reshape(-1)
yv = yv.reshape(-1)
mesh = np.concatenate((xv, yv))





min_score = 0

es = cma.CMAEvolutionStrategy(mesh, 3)

while not es.stop():
    solutions = es.ask()


    solutions_int = np.array(solutions, dtype=np.int32)

    scores = []
    for s in solutions_int:
        xv_dist = s[:len(xv)]
        yv_dist = s[len(xv):]

        transformation = np.array([xv, yv, xv_dist, yv_dist]).transpose()

        np.savetxt("{path}/points.txt".format(path=path), transformation, fmt='%i')

        subprocess.run('convert {path}/segment.png -virtual-pixel gray '
                       '-distort polynomial "3 $(cat {path}/points.txt)" '
                       '{path}/segment_distord.png'.format(path=path), shell=True)

        segment_distord = Sample("../data/tmp/segment_distord.png").get_image()

        segment_distord[segment_distord < 128] = 0
        segment_distord[segment_distord >= 128] = 255

        score = -(rot_grain == segment_distord).sum()

        if score < min_score:
            print(score)
            min_score = score

        fig = plt.figure(figsize=(15, 8))
        plt.imshow(Sample("../data/tmp/segment_distord.png").get_image(), interpolation='nearest', cmap=cm.gray)
        plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig("../data/tmp/out3.png")

        scores.append(score)

    es.tell(solutions, scores)
    es.disp()

es.result_pretty()





# Best score sans distorsion 1577 1.75 (-10, 122)

#('---> ', (1686, (-10, 121), 1.75, [0.0001343451017325502, -3.697271222337517e-07, -1.5041346294055493e-06, 3.445295683812902e-07], array([[  17.59044838,    0.        ,  296.37246704],
#       [   0.        ,   17.59044838,  302.71298218],



# for angle in np.arange(0, 4.0, 0.33):
#
#     M = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
#     rot_grain = cv2.warpAffine(grain, M, (height, width))
#
#     rot_grain[rot_grain < 128] = 0
#     rot_grain[rot_grain >= 128] = 255
#
#     for tx in range(-25, 5, 1):
#         for ty in range(105, 135, 1):
#
#             M = np.float32([[1, 0, tx], [0, 1, ty]])
#             segment_translate = cv2.warpAffine(segment, M, (height, width))
#
#             rot_grain[rot_grain < 128] = 1
#             rot_grain[rot_grain >= 128] = 255  # put a different background
#
#             score = (rot_grain == segment_translate).sum()
#
#             if score < 350:
#                 raise ScoreToLow
#
#             if score > max_score:
#                 max_score = score
#                 best_grain, best_segment = rot_grain, segment_translate
#                 best_val = (max_score, (tx, ty), angle)
#
#                 logger.info(("---> ", best_val))
#
#                 fig = plt.figure(figsize=(15, 8))
#                 plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
#                 plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
#                 fig.savefig("out-{}.png".format(score))


# logger.info("")
# logger.info("BEST SCORE:")
# logger.info(("###> ", best_val))

fig = plt.figure(figsize=(15, 8))
plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
fig.savefig("../data/tmp/out.png")

fig = plt.figure(figsize=(15, 8))
plt.imshow(Sample("../data/tmp/segment_distord.png").get_image(), interpolation='nearest', cmap=cm.gray)
plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
fig.savefig("../data/tmp/out3.png")

# link to fix distorsion:
# https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image



# ('---> ', (1674, (-11, 121), 1.3200000000000001, [4.832664937699144e-05, -1.2391147016077096e-08, 6.71011746205243e-05, 2.797055575188841e-07],
# array([[  19.43749428,    0.        ,  300.43862915],
#        [   0.        ,   19.43749428,  300.11886597],
#        [   0.        ,    0.        ,    1.        ]], dtype=float32)))

# ('---> ', (1710, (-10, 121), 1.6500000000000001, [-2.0077368897377396e-07, -1.2096681659557517e-08, 0.0002465578925599495, 8.566818843153823e-05],
# array([[  17.10153389,    0.        ,  315.25561523],
#        [   0.        ,   17.10153389,  308.22003174],
#        [   0.        ,    0.        ,    1.        ]], dtype=float32)))


#('---> ', (1711, (-10, 121), 1.75, [4.775150158382109e-05, -2.295702562245182e-08, -3.81355490560222e-07, 0.0001528079179168962],
# array([[  16.07833672,    0.        , 298.13970947],
#       [   0.        ,   16.07833672,  305.46511841],
#       [   0.        ,    0.        ,    1.        ]], dtype=float32)))