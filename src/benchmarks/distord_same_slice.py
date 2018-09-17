import subprocess
import os
import numpy as np
from sample import Sample

from argparse import Namespace

from matching import distord_segment

from matching import matching_tools as mt


args = Namespace(

    seg_ref_path="../../data/segment.align/segment.align.70.png",
    grain_ref_path="../../data/grain/Slice70_distor.bmp",
    out_dir="../../data/tmp/",
    tmp_dir="../../data/tmp/",

    no_points=30,
    std_pixels=5,
    polynom=3,

    max_sampling=3000,

    no_thread=2
)

x_to_crop = 110  # as segment are out of the score, remove useless grain to compute score

x_time = 10
scores, mesh, segments = [], [], []

for _ in range(x_time):

    # Compute distord segment
    _, segment, transformation = distord_segment.__main__(args)
    segments.append(segment)

    # Load grain
    grain = Sample(args.grain_ref_path).get_image()
    grain = np.invert(grain)

    grain[grain < 128] = 1  # force non-grain values to have a different value than segment
    grain[grain >= 128] = 255

    # crop useless grain
    segment = segment[x_to_crop:, :]
    grain = grain[x_to_crop:, :]

    # compute score after cropping useless parts of grain/segment
    score = mt.compute_score(segment=segment, ebsd=grain)
    scores.append(score)

    mesh.append(transformation)



useful_mesh = []
for m in mesh:
    new_mesh = []
    for pt in m:
        if pt[0] > x_to_crop:
            new_mesh.append(pt[2:])
    useful_mesh.append(new_mesh)

useful_mesh = np.array(useful_mesh)
useful_mesh = useful_mesh.reshape([useful_mesh.shape[0], -1])
mesh_std = useful_mesh.std(axis=0)

print("Mean score: ", np.mean(scores))
print("Std score: ", np.std(scores))

print("Mesh (mean of std): ", np.mean(mesh_std.mean()))
print("Mesh (std of std): ", np.mean(mesh_std.std()))

import pickle

data = dict()
data["mesh"] = mesh
data["scores"] = scores
data["grain_ref"] = Sample(args.grain_ref_path).get_image()
data["segment_ref"] = Sample(args.seg_ref_path).get_image()
data["segments"] = segments

with open("same_slice.pkl", "wb") as f:
    pickle.dump(data, f)







