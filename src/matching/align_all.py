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

    segment[segment < 128] = 0
    segment[segment >= 128] = 255

    # Load gains
    grain = Sample(args.grain_ref_path)#Sample("../data/grain/Slice70wDistorsions.bmp")
    grain = grain.get_image()
    grain = np.invert(grain) # we need the background to be black (default color for numpy transformation)

    grain[grain < 128] = 1
    grain[grain >= 128] = 255 # put a different background

    (width, height) = grain.shape
    best_translation, best_angle = None, None

    # Step 2: Look for best alignment
    print("Look for best alignment...")
    max_score = 0
    best_val = None
    for angle in range_angle:

        M_rot = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
        rot_segment = cv2.warpAffine(segment, M_rot, (height, width))

        for tx in range_translation_x:
            for ty in range_translation_y:

                M_aff = np.float32([[1, 0, tx], [0, 1, ty]])
                final_segment = cv2.warpAffine(rot_segment, M_aff, (height, width))

                final_segment[final_segment < 128] = 0
                final_segment[final_segment >= 128] = 255  # put a different background

                score = (grain == final_segment).sum()

                if score > max_score:
                    max_score = score
                    best_segment = final_segment
                    best_val = (max_score, (tx, ty), angle, M_rot, M_aff)
                    print(best_val)

                    fig = plt.figure(figsize=(15, 8))
                    plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
                    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
                    fig.savefig(os.path.join(args.tmp_dir, "ref_overlap-{}.png".format(score)))

            break
        break


    (max_score, (tx, ty), angle, fM_rot, fM_aff) = best_val
    print("----------------------------------------------")
    print("Best score: ",  max_score)
    print(" - tx  :  ", tx)
    print(" - ty  :  ", ty)
    print(" - rot : ", angle)
    print("----------------------------------------------")

    print("Process all slices...")
    # Step 3: crop and align images
    for segment_filename, grain_filename in zip(os.listdir(args.seg_dir), os.listdir(args.grain_dir)):

        id_segment = re.findall(r'\d+', segment_filename)[0]
        id_grain = re.findall(r'\d+', grain_filename)[0]

        assert id_grain == id_segment, "Mismatch between file's id : {} vs {}".format(segment_filename, grain_filename)

        # Load grain file
        grain_file = os.path.join(args.grain_dir, grain_filename)
        grain = Sample(grain_file).get_image()

        # Load segment file
        segment_file = os.path.join(args.seg_dir, segment_filename)
        segment = Sample(segment_file).get_image()


        # resize
        if rescale_x != 1 and rescale_y != 1:
            segment = cv2.resize(segment, None,
                                 fx=rescale_x,
                                 fy=rescale_y, interpolation=cv2.INTER_AREA)

        # Apply affine transformation
        rot_segment = cv2.warpAffine(segment, fM_rot, segment.shape) #Do not reduce the size to avoid loosing information
        final_segment = cv2.warpAffine(rot_segment, fM_aff, (height, width))

        # dump overlap for visual interpretation
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(final_segment, interpolation='nearest', cmap=cm.gray)
        plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig(os.path.join(args.tmp_dir, "ref_overlap-{}.png".format(id_grain)))

        filename_out = os.path.join(args.out_dir, "segment.align.{}.png".format(id_grain))
        cv2.imwrite(filename_out, final_segment)

    print("Done!")
