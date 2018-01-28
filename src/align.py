

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