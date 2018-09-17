import matching.distord_segment as distord

import argparse
import itertools

std_pixels = [3, 5, 7, 10, 15]
no_points_step = [50, 75, 100]
max_sampling = 3000
polynom = [3]

best_score = 0

for (std, step, p) in itertools.product(*[std_pixels, no_points_step, polynom]):

    print(std, step, p)

    args = argparse.Namespace(
        seg_ref_path="../../data/AM718/segment.align.718.png",
        grain_ref_path="../../data/AM718/AM718_speckle_straight.tif",
        out_dir="../../data/AM718/tmp",
        tmp_dir="../../data/AM718/tmp",

        no_points=None,
        no_points_step=step,
        std_pixels=std,
        max_sampling=max_sampling,
        polynom=p,

        invert_grain=True,
    )

    score, _, _ = distord.__main__(args)

    if score > best_score:
        best_score = score
        print(best_score, (std, step, p))

