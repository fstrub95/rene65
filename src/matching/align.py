import numpy as np
import io
import subprocess
import multiprocessing
import re
import os
import cv2

from sample import Sample
from matplotlib import pyplot as plt, cm

import argparse

if __name__ == "__main__":

    # Load conf
    from matching.align_conf import *

    parser = argparse.ArgumentParser('Image align input!')

    parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_ref_path", type=str, required=True, help="Path to the grain image")

    parser.add_argument("-seg_dir", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_dir", type=str, required=True, help="Path to the grain image")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory to store intermediate results")
    parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

    args = parser.parse_args()

    # Look for the ref segment/grain
    segment = Sample(args.seg_ref_path)  # Sample("../data/grain/Slice70wDistorsions.bmp")
    segment = segment.get_image()

    if precrop is not None:
        segment = segment[precrop.x_min:precrop.y_max, precrop.y_min:precrop.y_max]

    if rescale_x != 1 and rescale_y != 1:
        segment = cv2.resize(segment, None,
                             fx=rescale_x,
                             fy=rescale_y, interpolation=cv2.INTER_AREA)

    # Load gains
    grain = Sample(args.grain_ref_path)#Sample("../data/grain/Slice70wDistorsions.bmp")
    grain = grain.get_image()

    grain[grain < 128] = 1
    grain[grain >= 128] = 255  # put a different background

    (width, height) = grain.shape
    best_translation, best_angle = None, None

    # Step 2: Look for best alignment
    max_score = 0
    for angle in range_angle:

        M = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
        rot_segment = cv2.warpAffine(grain, M, (height, width))

        for tx in range_translation_x:
            for ty in range_translation_y:

                M = np.float32([[1, 0, tx], [0, 1, ty]])
                final_segment = cv2.warpAffine(rot_segment, M, (height, width), borderValue=255)

                final_segment[final_segment < 128] = 0
                final_segment[final_segment >= 128] = 255  # put a different background

                score = (grain == final_segment).sum()

                if score > max_score:
                    max_score = score
                    best_segment = final_segment
                    best_val = (max_score, (tx, ty), angle)

                    fig = plt.figure(figsize=(15, 8))
                    plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
                    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
                    fig.savefig(os.path.join(args.tmp_dir, "ref_overlap-{}.png".format(score)))

    # Step 3: crop and align images
    for sample_filename in os.listdir(args.img_dir):

        grain_file = os.path.join(args.img_dir, sample_filename)
        segment_file = os.path.join(args.seg_dir, sample_filename)

        grain = Sample(grain_file).get_image()
        segment = Sample(segment).get_image()

        fig = plt.figure(figsize=(15, 8))
        plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
        plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig(os.path.join(args.tmp_dir, "ref_overlap-{}.png".format(score)))

        fig = plt.figure(figsize=(15, 8))
        plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
        plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig("out-{}.png".format(score))
