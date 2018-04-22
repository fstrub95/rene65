import matching.distord as distord

import argparse
from misc.pickle_helper import pickle_dump
import numpy as np
from sample import Sample

from sklearn.metrics import jaccard_similarity_score

no_points_step = 75
std_pixels = 7
max_sampling = 3000
polynom = 3


scores = []
segments = []
transformations = []

for i in range(0, 10):

    args = argparse.Namespace(
        seg_ref_path="../../data/AM718/segment.align.718.png",
        grain_ref_path="../../data/AM718/AM718_speckle_straight.tif",
        out_dir="../../data/AM718/tmp/{}/".format(i),
        tmp_dir="../../data/AM718/tmp/{}/".format(i),

        no_points=None,
        no_points_step=no_points_step,
        std_pixels=std_pixels,
        max_sampling=max_sampling,
        polynom=polynom,
    )

    best_score, s, t = distord.__main__(args)

    scores.append(best_score)
    segments.append(s)
    transformations.append(t)

pickle_dump(
    dict(scores=scores, segments=segments, transformations=transformations),
    "../../data/AM718/std.pkl"
)

print("Score: {} +/- {}".format(np.mean(scores), np.std(scores)))

x_std = np.array(transformations)[:,:,2].std(axis=0)
y_std = np.array(transformations)[:,:,3].std(axis=0)


print("tx: {} +/- {}".format(np.mean(x_std), np.std(x_std)))
print("ty: {} +/- {}".format(np.mean(y_std), np.std(y_std)))

segment = Sample("../../data/AM718/segment.align.718.png").get_image()

jacard_distance_align_distord = []
for s in segments:
    jacard_distance_align_distord += [jaccard_similarity_score(
        segment.ravel(),
        s.ravel())]

print("jacard_distance_align_distord: {} +/- {}".format(np.mean(jacard_distance_align_distord), np.std(jacard_distance_align_distord)))

jacard_distance = []
for i, s1 in enumerate(segments):
    for j, s2 in enumerate(segments):
        if i != j:
            jacard_distance += [jaccard_similarity_score(
                s1.ravel(),
                s2.ravel())]

print("jacard_distance: {} +/- {}".format(np.mean(jacard_distance), np.std(jacard_distance)))
