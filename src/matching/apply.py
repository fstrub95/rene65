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

    parser.add_argument("-seg_dir", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_dir", type=str, required=True, help="Path to the grain image")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory to output segment.align")

    parser.add_argument("-affine_json", type=str, default=None, help="Path to the affine transformation")
    parser.add_argument("-mesh_file", type=str, default=None, help="Path to the mesh file from align")
    parser.add_argument("-polynom", type=int, default=3, help="Polynom used to distord the image")

    parser.add_argument("-invert_segment", type=bool, default=False, help="Put True if background is white")
    parser.add_argument("-invert_grain", type=bool, default=False, help="Put True if background is white")

    args = parser.parse_args()

    ######
    # Step 1 : Align
    #####

    # Look for the ref segment
    segment = Sample(args.seg_ref_path)
    segment = segment.get_image()

    if args.invert_segment:
        segment = np.invert(segment)  # we need the background to be black (default color for numpy transformation)

        segment[segment < 128] = 0
        segment[segment >= 128] = 255

    # Load gains
    grain = Sample(args.grain_ref_path)
    grain = grain.get_image()

    if args.invert_grain:
        grain = np.invert(grain)  # we need the background to be black (default color for numpy transformation)

    grain[grain < 128] = 1
    grain[grain >= 128] = 255  # put a different background

    ######
    # Step 1 : Align
    #####

    if args.affine_json is not None:
        with open(args.affine_json, "r") as f:
            data = json.load(f)

        precop = data.get("precrop", None)
        [rescale_x, rescale_y] = data["rescale"]
        [tx, ty] = data["translate"]
        angle = data["angle"]

        aligner = mt.Aligner(precrop=precrop,
                             rescale=(rescale_x, rescale_y))

        align_segment, score = aligner.apply(
            grain=grain, segment=segment,
            tx=tx, ty=ty,
            angle=angle)

        score = mt.compute_score(segment=align_segment, ebsd=grain)
        print("score after alignment: {}".format(score))

    ######
    # Step 2 : Distord
    ####

    if args.mesh_file is not None:
        mesh = []
        with open(args.mesh_file, "r") as file:
            for l in file.readline():
                mesh += [l.split(sep=" ")]
        mesh = np.array(mesh).transpose()

        segment_distord = mt.apply_distortion(segment=segment,
                                              points=mesh,
                                              polynom=args.polynom)

        score = mt.compute_score(segment=segment_distord, ebsd=grain)
        print("score after distord: {}".format(score))








