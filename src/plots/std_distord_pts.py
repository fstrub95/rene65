import matching.distord as distord

import argparse
from misc.pickle_helper import pickle_dump, pickle_loader
import numpy as np
from sample import Sample

from sklearn.metrics import jaccard_similarity_score
import os

from matching import matching_tools as mt

no_points_step = 75
std_pixels = 5
max_sampling = 2000
polynom = 3
root = "/home/fstrub/Projects/rene65.backup/data/rene65/"
# root = "/home/fstrub/Projects/rene65.backup/data/rene65/"

scores = []
segments = []
transformations = []
all_scores = []

# for i in range(0, 100):
#
#     print("Start run {}".format(i))
#
#     out_dir = root + "tmp/{}/".format(i)
#
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     args = argparse.Namespace(
#         seg_ref_path=root + "segment.align/segment.align.png",
#         ebsd_ref_path=root + "grain/R65_ebsd.png",
#         out_dir=out_dir,
#         tmp_dir=out_dir,
#
#         no_points=None,
#         no_points_step=no_points_step,
#         std_pixels=std_pixels,
#         max_sampling=max_sampling,
#         polynom=polynom,
#
#         invert_grain=False,
#     )
#
#     best_score, s, t, a_s = distord.__main__(args)
#
#     scores.append(best_score)
#     segments.append(s)
#     transformations.append(t)
#     all_scores.append(a_s)
#
#     print("dumping...")
#
#     pickle_dump(
#         dict(scores=scores,
#              segments=segments,
#              transformations=transformations,
#              all_scores=all_scores),
#         root + "std.pkl"
#     )


data = pickle_loader("/home/fstrub/Projects/rene65.backup/out/std.pkl")
scores = data["scores"]
segments = data["segments"]
transformations = data["transformations"]
all_scores = data["all_scores"]

print(np.array(scores).shape)

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt, cm
fig = plt.figure(figsize=(15, 8))
sns.heatmap(np.array(segments).mean(axis=0) / 255,   yticklabels=False, xticklabels=False)
fig.savefig(root + "confidence.png")


print("Score: {} +/- {}".format(np.mean(scores), np.std(scores)))

x_std = np.array(transformations)[:,:,2].std(axis=0)
y_std = np.array(transformations)[:,:,3].std(axis=0)


print("tx: {} +/- {}".format(np.mean(x_std), np.std(x_std)))
print("ty: {} +/- {}".format(np.mean(y_std), np.std(y_std)))

segment = Sample(root + "segment.align/segment.align.png").get_image()

f1_distance_align_distord = []
for s1 in segments:
    s = np.copy(s1)
    s[s1 < 128] = 1  # force non-grain values to have a different value than segment
    s[s1 >= 128] = 255
    f1_distance_align_distord += [mt.compute_score(segment=segment, grain=s)]

print("f1_distance_align_distord: {} +/- {}".format(
    np.mean(f1_distance_align_distord), np.std(f1_distance_align_distord)))

f1_distance = []
for i, s1 in enumerate(segments):
    for j, s2 in enumerate(segments):
        if i < j:
            s = np.copy(s1)
            s[s1 < 128] = 1  # force non-grain values to have a different value than segment
            s[s1 >= 128] = 255
            f1_distance += [mt.compute_score(segment=s, grain=s2)]

print("f1_distance: {} +/- {}".format(np.mean(f1_distance), np.std(f1_distance)))



segment = Sample(root + "segment.align/segment.align.png").get_image()
jacard_distance_align_distord = []
for s in segments:
    jacard_distance_align_distord += [jaccard_similarity_score(
        segment.ravel(),
        s.ravel())]

print("jacard_distance_align_distord: {} +/- {}".format(np.mean(jacard_distance_align_distord), np.std(jacard_distance_align_distord)))

jacard_distance = []
for i, s1 in enumerate(segments):
    for j, s2 in enumerate(segments):
        if i < j:
            jacard_distance += [jaccard_similarity_score(
                s1.ravel(),
                s2.ravel())]

print("jacard_distance: {} +/- {}".format(np.mean(jacard_distance), np.std(jacard_distance)))


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

all_scores = np.array(all_scores)
mean = all_scores.mean(axis=0)[:max_sampling]
std = all_scores.std(axis=0)[:max_sampling]

mean = moving_average(mean)
std = moving_average(std)

t = range(len(mean))
lower_bound = mean + std
upper_bound = mean - std



fig, ax = plt.subplots(1)
ax.plot(t, mean, color='blue')
ax.fill_between(t, lower_bound, upper_bound, facecolor='blue', alpha=0.4)
ax.set_xlim(0, max_sampling)
fig.savefig(root + "training.png")
