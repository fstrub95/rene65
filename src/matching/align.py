import os
import cv2
import json
import re

from sample import Sample
from matplotlib import pyplot as plt, cm

import argparse

from matching import matching_tools as mt

# Load external conf
from matching.align_conf import *


def __main__(args=None):

    if args is None:

        parser = argparse.ArgumentParser('Image align input!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-grain_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-align_dir", type=str, required=True, help="Output directory (segment crop)")
        parser.add_argument("-out_dir", type=str, required=True, help="Output directory (image overlap + affine.pkl)")

        parser.add_argument("-invert_segment", type=bool, default=False, help="Put True if background is white")
        parser.add_argument("-invert_grain", type=bool, default=False, help="Put True if background is white")


        args = parser.parse_args()

    id_segment = re.findall(r'\d+', os.path.basename(args.seg_ref_path))[0]
    id_grain = re.findall(r'\d+', os.path.basename(args.grain_ref_path))[0]

    assert id_grain == id_segment, "Mismatch between file's id : {} vs {}".format(args.seg_ref_path, args.grain_ref_path)

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

    normalization_score = (grain == 255).sum()

    # Step 2: Look for best alignment
    print("Look for best alignment...")
    best_score = 0
    best_val, best_segment = None, None

    for angle in range_angle:

        center_rotation = (segment.shape[0] / 2, segment.shape[1] / 2)
        M_rot = cv2.getRotationMatrix2D(center_rotation, angle, 1)
        rot_segment = cv2.warpAffine(segment, M_rot, segment.shape[::-1])

        # Note: we perform the precrop after rotate
        if precrop is not None:
            rot_segment = rot_segment[precrop.x_min:precrop.x_max, precrop.y_min:precrop.y_max]

        if rescale_x != 1 and rescale_y != 1:
            rot_segment = cv2.resize(rot_segment, None,
                                 fx=rescale_x,
                                 fy=rescale_y, interpolation=cv2.INTER_AREA)

        for tx in range_translation_x:
            for ty in range_translation_y:

                M_aff = np.float32([[1, 0, tx], [0, 1, ty]])
                align_segment = cv2.warpAffine(rot_segment, M_aff, grain.shape[::-1])

                align_segment[align_segment < 128] = 0
                align_segment[align_segment >= 128] = 255

                score = mt.compute_score(segment=align_segment, grain=grain, normalization=normalization_score)

                if score > best_score:
                    best_score = score
                    best_segment = align_segment
                    best_val = (best_score, (tx, ty), angle)
                    print("Score: {0:.4f}, tx: {1}, ty: {2}, angle: {3}".format(float(best_score), tx, ty, angle))


    # Display results
    (best_score, (tx, ty), angle) = best_val
    print("----------------------------------------------")
    print("Best score: ", best_score)
    print(" - tx  :  ", tx)
    print(" - ty  :  ", ty)
    print(" - rot : ", angle)
    print("----------------------------------------------")

    # Plot crop/align segment
    filename_out = os.path.join(args.align_dir, "segment.align.{}.png".format(id_grain))
    cv2.imwrite(filename_out, best_segment)

    # Plot overlap visualization
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(os.path.join(args.out_dir, "ref_overlap-{0}.png".format(id_grain)))

    # Dump affine.pkl with all the required information to perform the affine transformation
    data = dict()
    with open(os.path.join(args.out_dir, "affine.{}.json".format(id_grain)) , "wb") as f:
        data["id_slice"] = id_grain
        data["rescale"] = [rescale_x, rescale_y]
        data["translate"] = [tx, ty]
        data["angle"] = angle

        if precrop is not None:
            data["precrop"] = { "x_min" : precrop.x_min,  "x_max": precrop.x_max, "y_min": precrop.y_min, "y_max": precrop.y_max}

        results_json = json.dumps(data)
        f.write(results_json.encode('utf8', 'replace'))

    return best_score, best_segment, data

if __name__ == "__main__":
    __main__()
