import numpy as np
import cv2
from sample import Sample
from matplotlib import pyplot as plt, cm

def show_img(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img)
    cv2.waitKey(0)

min_row = 0
max_row = 1565
min_col = 700
max_col = 2715


segment = Sample("../data/segment/Slice70.tif")
segment = segment.get_image()
segment = segment[min_row:max_row, min_col:max_col]
segment = cv2.resize(segment, None, fx=0.3265, fy=0.3265, interpolation = cv2.INTER_AREA)



grain = Sample("../data/grain_size/Slice70wDistorsions.bmp")
grain = grain.get_image()


(rows, cols) = grain.shape
max_score = 0
best_translation, best_angle = None, None
best_grain, best_segment = None, None


for angle in np.arange(0.0, 3.0, 0.25):

    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rot_grain = cv2.warpAffine(grain, M, (cols, rows), borderValue=255)

    rot_grain[rot_grain < 128] = 0
    rot_grain[rot_grain >= 128] = 255

    print("new_angle:", angle)

    for tx in range(-50, 50, 1):
        for ty in range(100, 150, 1):


            M = np.float32([[1, 0, tx], [0, 1, ty]])
            segment_translate = cv2.warpAffine(segment, M, (cols, rows), borderValue=255)

            rot_grain[rot_grain < 128] = 0
            rot_grain[rot_grain >= 128] = 254  # put a different background

            score = (rot_grain == segment_translate).sum()
            if score > max_score:
                max_score = score
                best_translation, best_angle = angle, (tx, ty)
                best_grain, best_segment = rot_grain, segment_translate
                print(max_score, best_translation, best_angle)

                fig = plt.figure(figsize=(15, 8))
                plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
                plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
                fig.savefig("out.png")

print(max_score, best_translation, best_angle)

fig = plt.figure(figsize=(15, 8))
plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
plt.imshow(best_grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
fig.savefig("out.png")

# link to fix distorsion:
# https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image






