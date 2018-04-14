import json
import re
import os
import cv2

from sample import Sample
from matplotlib import pyplot as plt, cm
from matching import matching_tools as mt


import argparse

if __name__ == "__main__":

    # Load conf
    from matching.align_conf import *

    parser = argparse.ArgumentParser('Image align input!')

    parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_ref_path", type=str, required=True, help="Path to the grain image")

    parser.add_argument("-out_dir", type=str, required=True, help="Directory to output segment.align")
    parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

    parser.add_argument("-affine_json", type=str, required=True, help="Path to the affine transformation")

    args = parser.parse_args()

    with open(args.affine_json, "r") as f:
        data = json.load(f)

    precop = data.get("precrop", None)
    [rescale_x, rescale_y] = data["rescale"]
    [tx, ty] = data["translate"]
    angle = data["angle"]

    print("----------------------------------------------")
    print("Transformation: ")
    print(" - tx  :  ", tx)
    print(" - ty  :  ", ty)
    print(" - rot : ", angle)
    print("----------------------------------------------")


    print("Process all slices...")
    coverage, n_slice = 0, 0
    # Step 3: crop and align images
    for segment_filename, grain_filename in zip(os.listdir(args.seg_dir), os.listdir(args.grain_dir)):

        id_segment = re.findall(r'\d+', segment_filename)[0]
        id_grain = re.findall(r'\d+', grain_filename)[0]

        assert id_grain == id_segment, "Mismatch between file's id : {} vs {}".format(segment_filename, grain_filename)

        # Load grain file
        grain_file = os.path.join(args.grain_dir, grain_filename)
        grain = Sample(grain_file).get_image()
        grain = np.invert(grain)  # we need the background to be black (default color for numpy transformation)

        grain[grain < 128] = 1
        grain[grain >= 128] = 255

        # Load segment file
        segment_file = os.path.join(args.seg_dir, segment_filename)
        segment = Sample(segment_file).get_image()

        aligner = mt.Aligner(precrop=precrop,
                             rescale=(rescale_x, rescale_y))

        # perform alignment
        final_segment, score = aligner.apply(
            grain=grain, segment=segment,
            tx=tx, ty=ty,
            angle=angle)

        print("Slice {0} : {1:.4f}".format(id_grain,score))

        # dump overlap for visual interpretation
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(final_segment, interpolation='nearest', cmap=cm.gray)
        plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig(os.path.join(args.tmp_dir, "overlap-{}.png".format(id_grain)))

        filename_out = os.path.join(args.out_dir, "segment.align.{}.png".format(id_grain))
        cv2.imwrite(filename_out, final_segment)

        n_slice += 1

    coverage /= n_slice

    print("Average grain coverage: {0:.4f}".format(coverage))
    print("Done!")
