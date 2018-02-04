# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                 #
# Copyright (c) 2017 Marie-Agathe Charpagne & Florian Strub                       #
# All rights reserved.                                                            #
#                                                                                 #
# Redistribution and use in source and binary forms, with or without              #
# modification, are permitted provided that the following conditions are met:     #
#     * Redistributions of source code must retain the above copyright            #
#       notice, this list of conditions and the following disclaimer.             #
#     * Redistributions in binary form must reproduce the above copyright         #
#       notice, this list of conditions and the following disclaimer in the       #
#       documentation and/or other materials provided with the distribution.      #
#     * Neither the name of the <organization> nor the                            #
#       names of its contributors may be used to endorse or promote products      #
#       derived from this software without specific prior written permission.     #
#                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY              #
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES      #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    #
#                                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse
import multiprocessing
import os
import logging
import numpy as np
from scipy import ndimage
import json
from misc.logger import create_logger
from crystallography.tsl import OimScan
from pymicro.view.vol_utils import compute_affine_transform
import re
from pymicro.external.tifffile import TiffFile
from sample import Sample
import copy

from matplotlib import pyplot as plt, cm



def parse_and_match(data):

    seg_in = data["seg_in"]
    ang_in = data["ang_in"]
    ang_out = data["ang_out"]
    iq_out = data["iq_out"]
    match_out = data["match_out"]

    ref_points = data["ref_points"]
    tsr_points = data["tsr_points"]

    slice_id = data["slice_id"]

    logger = logging.getLogger()
    logger.info("Processing file: {}".format(seg_in))


    scan = OimScan(ang_in)
    iq = scan.iq  # size is 601x601
    pixel_size = 0.5  # micron

    # compute the affine transform from the point set
    translation, transformation = compute_affine_transform(ref_points, tsr_points)
    invt = np.linalg.inv(transformation)
    offset = -np.dot(invt, translation)

    sem = TiffFile(seg_in).asarray()[:2048, :].T  # imaged with scaling already applied


    # Register new ebsd with transformation
    iq_reg = ndimage.interpolation.affine_transform(iq.T, invt, output_shape=sem.shape, offset=offset).T
    eu0_reg = ndimage.interpolation.affine_transform(scan.euler[:,:, 0].T, invt, output_shape=sem.shape, offset=offset, order=0).T
    eu1_reg = ndimage.interpolation.affine_transform(scan.euler[:,:, 1].T, invt, output_shape=sem.shape, offset=offset, order=0).T
    eu2_reg = ndimage.interpolation.affine_transform(scan.euler[:,:, 2].T, invt, output_shape=sem.shape, offset=offset, order=0).T
    ci_reg = ndimage.interpolation.affine_transform(scan.ci.T, invt, output_shape=sem.shape, offset=offset, order=0, cval=-1).T


    # Crop big .ang file
    # Calculate coordinates of the estimated crop
    # x = compositeScan.iq
    # sum_cols = np.sum(x,axis=0)
    # sum_rows = np.sum(x,axis=1)
    # in_rows = np.nonzero(sum_rows)
    # in_cols = np.nonzero(sum_cols)
    # min_row = np.min(in_rows)
    # max_row = np.max(in_rows)
    # min_col = np.min(in_cols)
    # max_col = np.max(in_cols)
    min_row = 0
    max_row = 1565
    min_col = 700
    max_col = 2715


    sem = sem.T[min_row:max_row, min_col:max_col]  # imaged with scaling already applied
    compositeScan = OimScan.zeros_like(sem, resolution=(pixel_size, pixel_size))
    compositeScan.sampleId = 'R65_09_13_17_{}'.format(str(slice_id).zfill(3))

    compositeScan.euler[:, :, 0] = eu0_reg[min_row:max_row, min_col:max_col]
    compositeScan.euler[:, :, 1] = eu1_reg[min_row:max_row, min_col:max_col]
    compositeScan.euler[:, :, 2] = eu2_reg[min_row:max_row, min_col:max_col]
    compositeScan.iq = iq_reg[min_row:max_row, min_col:max_col]
    compositeScan.ci = ci_reg[min_row:max_row, min_col:max_col]
    compositeScan.sem = np.ones_like(ci_reg[min_row:max_row, min_col:max_col])

    # Create phase
    sample = Sample(seg_in)
    segment = sample.get_image()[min_row:max_row, min_col:max_col]
    phase = np.copy(sample.get_image())[min_row:max_row, min_col:max_col]
    phase[np.where(segment > 255 / 2)] = 1
    phase[np.where(segment <= 255 / 2)] = 2
    compositeScan.phase = phase

    # Match precipitate segmentation and EBSD
    phase1 = scan.phaseList[0]
    phase2 = copy.copy(scan.phaseList[0])
    phase2.number = 2
    phase2.formula = 'Ma'
    phase2.materialName = 'Matrix'
    compositeScan.phaseList = [phase1, phase2]



    fig = plt.figure(figsize=(15, 8))
    plt.imshow(segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(compositeScan.iq, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(match_out)

    compositeScan.writeAng(ang_out)

    plt.imsave(iq_out, compositeScan.iq, cmap=cm.gray)



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-seg_dir", type=str, required=True, help="Directory with input image")
    parser.add_argument("-seg_ext", type=str, default=".tif", help="Segmentation extension")
    parser.add_argument("-ang_dir", type=str, required=True, help="Directory with ang info")
    parser.add_argument("-iq_out_dir", type=str, required=True, help="Directory to output iq image")
    parser.add_argument("-ang_out_dir", type=str, required=True, help="Directory to output updated ang info")
    parser.add_argument("-no_thread", type=int, default=1, help="Number of thread to run execute the segmentation")
    parser.add_argument("-calibration", type=str, required=True, help="Path to the file containing the list of points")

    args = parser.parse_args()

    # create technical python tools
    logger = create_logger(os.path.join(args.seg_dir, "log.txt"))
    pool = multiprocessing.Pool(processes=args.no_thread)

    with open(args.calibration, "r") as file:
        sliceid_2_points = json.load(file)

    # Prepare sample for parallel computation
    data = []
    for seg_filename in os.listdir(args.seg_dir):

        if seg_filename.endswith(args.seg_ext):

            # Retrieve slice id
            res = re.findall(r'\d+', seg_filename)
            assert len(res) == 1, "Fail to retrieve the slide id in {}".format(seg_filename)
            slice_id = res[0]

            # Retrieve ang file
            ang_filename = "R65_09_13_17_{}_Mod.ang".format(str(slice_id).zfill(3)) # pad id with 0
            iq_filename = "iq_reg_{}.png".format(str(slice_id).zfill(3))
            match_filename = "match_iq_seg_{}.png".format(str(slice_id).zfill(3))

            d = {
                "seg_in": os.path.join(args.seg_dir, seg_filename),
                "ang_in": os.path.join(args.ang_dir, ang_filename),
                "iq_out": os.path.join(args.iq_out_dir, iq_filename),
                "match_out": os.path.join(args.iq_out_dir, match_filename),
                "ang_out": os.path.join(args.ang_out_dir, ang_filename),
                "ref_points": sliceid_2_points[slice_id]["ref_points"],
                "tsr_points": sliceid_2_points[slice_id]["tsr_points"],
                "slice_id": slice_id,
                }
            data.append(d)

    # execute multiprocessor script

    # for d in data:
    #     parse_and_match(d)
    pool.map(parse_and_match, data)

    logger.info("Done!")

