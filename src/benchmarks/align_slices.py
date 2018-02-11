import os
import numpy as np
from sample import Sample

from argparse import Namespace

from matching import align

from matching import matching_tools as mt
import re


x_to_crop = 110  # as segment are out of the score, remove useless grain to compute score

seg_dir = "../../data/segment"
grain_dir = "../../data/grain"
tmp_dir = "../../data/tmp"

scores = []
tx, ty, angle = [], [], []

for segment_filename, grain_filename in zip(os.listdir(seg_dir), os.listdir(grain_dir)):
    id_segment = re.findall(r'\d+', segment_filename)[0]
    id_grain = re.findall(r'\d+', grain_filename)[0]

    assert id_grain == id_segment, "Mismatch between file's id : {} vs {}".format(segment_filename, grain_filename)

    seg_ref_path = os.path.join(seg_dir, segment_filename)
    grain_ref_path = os.path.join(grain_dir, grain_filename)

    args = Namespace(
        seg_ref_path=seg_ref_path ,
        grain_ref_path=grain_ref_path,
        out_dir=tmp_dir,
    )

    # Compute distord segment
    _, segment, data = align.__main__(args)

    # Load grain
    grain = Sample(grain_ref_path).get_image()
    grain = np.invert(grain)
    grain[grain < 128] = 1  # force non-grain values to have a different value than segment
    grain[grain >= 128] = 255

    # crop useless grain
    segment = segment[x_to_crop:, :]
    grain = grain[x_to_crop:, :]

    # compute score after cropping useless parts of grain/segment
    score = mt.compute_score(segment=segment, grain=grain)
    scores.append(score)

    tx.append(data["translate"][0])
    ty.append(data["translate"][1])
    angle.append(data["angle"])

print("Mean score: ", np.mean(scores))
print("Std score: ", np.std(scores))

print("tx    : {0:.2f} +/- {1:.2f}".format(np.mean(tx), np.std(tx)))
print("ty    : {0:.2f} +/- {1:.2f}".format(np.mean(ty), np.std(ty)))
print("angle : {0:.2f} +/- {1:.2f}".format(np.mean(angle), np.std(angle)))






