import argparse
import multiprocessing
import os
import logging
import numpy as np
import pandas as pd
#import cv2
from scipy import ndimage
import json
from misc.logger import create_logger
from crystallography.tsl import OimScan
from sample import Sample
from pymicro.view.vol_utils import compute_affine_transform
import re
from pymicro.external.tifffile import TiffFile

# markers in Slice70.tif

def parse_and_match(data):


    segment_in = data["segment_in"]
    ang_in = data["ang_in"]
    ang_out = data["ang_out"]
    ref_points = data["ref_points"]
    tsr_points = data["tsr_points"]

    slice_id = data["slice_id"]

    logger = logging.getLogger()
    logger.info("Processing file: {}".format(segment_in))


    scan = OimScan(ang_in)
    iq = scan.iq  # size is 601x601
    pixel_size = 0.5  # micron

    # compute the affine transform from the point set
    translation, transformation = compute_affine_transform(ref_points, tsr_points)
    invt = np.linalg.inv(transformation)
    offset = -np.dot(invt, translation)


    sem = TiffFile('Slice{}.tif'.format(slice_id)).asarray()[:2048, :].T  # imaged with scaling already applied
    compositeScan = OimScan.zeros_like(sem.T, resolution=(pixel_size, pixel_size))
    compositeScan.sampleId = 'R65_09_13_17'.format(str(slice_id).zfill(3))

    # register ebsd
    iq_reg = ndimage.interpolation.affine_transform(iq.T, invt, output_shape=sem.shape, offset=offset).T
    eu0_reg = ndimage.interpolation.affine_transform(scan.euler[:, 0].T, invt, output_shape=sem.shape, offset=offset,
                                                     order=0).T
    eu1_reg = ndimage.interpolation.affine_transform(scan.euler[:, 1].T, invt, output_shape=sem.shape, offset=offset,
                                                     order=0).T
    eu2_reg = ndimage.interpolation.affine_transform(scan.euler[:, 2].T, invt, output_shape=sem.shape, offset=offset,
                                                     order=0).T
    ci_reg = ndimage.interpolation.affine_transform(scan.ci.T, invt, output_shape=sem.shape, offset=offset, order=0,
                                                    cval=-1).T
    ph_reg = ndimage.interpolation.affine_transform(scan.phase.T, invt, output_shape=sem.shape, offset=offset,
                                                    order=0).T
    compositeScan.euler[:, :, 0] = eu0_reg
    compositeScan.euler[:, :, 1] = eu1_reg
    compositeScan.euler[:, :, 2] = eu2_reg
    compositeScan.phase = ph_reg
    compositeScan.ci = ci_reg
    compositeScan.iq = iq_reg

    compositeScan.writeAng(ang_out)

    plt.imsave('iq_reg_070.png', compositeScan.iq, cmap=cm.gray)
    plt.imsave('sem_070.png', sem.T, cmap=cm.gray)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-seg_dir", type=str, required=True, help="Directory with segmentation")
    parser.add_argument("-seg_ext", type=str, default=".tif", help="Segmentation extension")
    parser.add_argument("-ang_dir", type=str, required=True, help="Directory with ang info")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory to output updated ang info")
    parser.add_argument("-no_thread", type=int, default=2, help="Number of thread to run execute the segmentation")
    parser.add_argument("-json", type=str, required=True, help="Path to the file containing the list of points")

    args = parser.parse_args()

    # create technical python tools
    logger = create_logger(os.path.join(args.seg_dir, "log.txt"))
    pool = multiprocessing.Pool(processes=args.no_thread)

    with open(args.json, "r") as file:
        sliceid_2_points = json.load(file)

    # Prepare sample for parallel computation
    data = []
    for segment_filename in os.listdir(args.seg_dir):

        if segment_filename.endswith(args.seg_ext):

            # Retrieve slice id
            res = re.findall(r'\d+', segment_filename)
            assert len(res) == 1, "Fail to retrieve the slide id in {}".format(segment_filename)
            slice_id = res[0]

            # Retrieve ang file
            ang_filename = "R65_09_13_17_{}_Mod.ang".format(str(slice_id).zfill(3)) # pad id with 0

            d = {
                "segment_in": os.path.join(args.seg_dir, segment_filename),
                "ang_in": os.path.join(args.ang_dir, ang_filename),
                "ang_out": os.path.join(args.out_dir, ang_filename),
                "slice_id": slice_id,
                "ref_points": sliceid_2_points[slice_id]["ref_points"],
                "tsr_points": sliceid_2_points[slice_id]["tsr_points"],
                }
            data.append(d)

    # execute multiprocessor script

    # for d in data:
    #     parse_and_match(d)
    pool.map(parse_and_match, data)

    logger.info("Done!")

