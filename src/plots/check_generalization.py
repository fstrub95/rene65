import matching.distord as distord

import argparse
from matching import matching_tools as mt
from sample import Sample
import numpy as np

no_points = 25
std_pixels = 7
max_sampling = 2500
polynom = 3


seg_ref_path_train = "../../data/AM718/segment.align.718.train.png"
seg_ref_path_eval = "../../data/AM718/segment.align.718.eval.png"
grain_ref_path = "../../data/AM718/AM718_speckle_straight.tif"

args = argparse.Namespace(
    seg_ref_path=seg_ref_path_train,
    grain_ref_path=grain_ref_path,
    out_dir="../../data/AM718/tmp",
    tmp_dir="../../data/AM718/tmp",

    no_points=no_points,
    no_points_step=None,
    std_pixels=std_pixels,
    max_sampling=max_sampling,
    polynom=polynom,
)


score, train_distord_segment, transformation = distord.__main__(args)

### EVAL


train_align_segment = Sample(seg_ref_path_train).get_image()
eval_align_segment = Sample(seg_ref_path_eval).get_image()

# Load grain
grain = Sample(grain_ref_path).get_image()
grain = np.invert(grain)

grain[grain < 128] = 1  # force non-grain values to have a different value than segment
grain[grain >= 128] = 255

eval_distord_segment = mt.apply_distortion(segment=eval_align_segment,
                                    polynom=polynom,
                                    points=transformation,
                                    segment_path_out="../../data/AM718/eval.generalization.png")



train_ratio_grain = mt.compute_score(segment=train_distord_segment, grain=grain)
eval_ratio_grain = mt.compute_score(segment=eval_distord_segment, grain=grain)
assert(score == train_ratio_grain)


train_ratio_segment = mt.compute_score(segment=grain, grain=train_distord_segment)
eval_ratio_segment = mt.compute_score(segment=grain, grain=eval_distord_segment)

print(train_ratio_grain, eval_ratio_grain)
print(train_ratio_segment, eval_ratio_segment)

# Plot how grain/segment overlap
from matplotlib import pyplot as plt, cm
fig = plt.figure(figsize=(15, 8))
plt.imshow(eval_distord_segment, interpolation='nearest', cmap=cm.gray)
plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
fig.savefig("../../data/AM718/eval.generalization.overlap.png")

