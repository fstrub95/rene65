import argparse
import multiprocessing
import os
import logging
import numpy as np
import pandas as pd
import cv2

from misc.logger import create_logger

from sample import Sample

import re


INDEX_SEGMENTATION = 8

def extract_headers(ang_file):
    '''
    Function to extract the size and step size of the input image
    Note that the computation is not optimal, but it pretty easy to update and maintain
    '''
    with open(ang_file, 'r') as fobj:

        # extract headers
        headers = ""
        for line in fobj:
            if line.startswith("#"):
                headers += line
            else:
                break

    # extract info from headers (we assume that data always exist...)
    xstep = float(re.search(r'# XSTEP: (\d+\.\d+)', headers).group(1))
    ystep = float(re.search(r'# YSTEP: (\d+\.\d+)', headers).group(1))
    ncols = int(re.search(r'# NCOLS_ODD: (\d+)', headers).group(1))
    nrows = int(re.search(r'# NROWS: (\d+)', headers).group(1))

    return (xstep, ystep, ncols, nrows), headers

def parse_and_match(data):

    segment_in = data["segment_in"]
    ang_in = data["ang_in"]
    ang_out = data["ang_out"]
    coord = data["coordinates"]


    logger = logging.getLogger()
    logger.info("Processing file: {}".format(segment_in))

    # Load image
    sample = Sample(segment_in)

    # Parse ang
    ang = pd.read_csv(ang_in, comment='#', sep="\s+", header=None)
    (xstep, ystep, ncols, nrows), header = extract_headers(ang_in)


    # Fit segment to ang
    segment = sample.get_image()

    crop = segment[coord["x1"]:coord["x2"], coord["y1"]:coord["y2"]]
    crop = cv2.resize(crop, (nrows , ncols), interpolation=cv2.INTER_NEAREST)
    crop = crop.reshape(-1)

    crop[np.where(crop >  255 / 2)] = 255
    crop[np.where(crop <= 255 / 2)] = 0

    ang.iloc[:, INDEX_SEGMENTATION] = crop

    # dump new ang
    logger.info("Dumping file: {}".format(ang_out))
    with open(ang_out, 'w') as fobj:
        fobj.write(header)
        ang.to_csv(fobj, sep='\t', encoding='utf-8', header=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-seg_dir", type=str, required=True, help="Directory with segmentation")
    parser.add_argument("-seg_ext", type=str, default=".tif", help="Segmentation extension")
    parser.add_argument("-ang_dir", type=str, required=True, help="Directory with ang info")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory to output updated ang info")
    parser.add_argument("-no_thread", type=int, default=2, help="Number of thread to run execute the segmentation")

    args = parser.parse_args()

    # create technical python tools
    logger = create_logger(os.path.join(args.seg_dir, "log.txt"))
    pool = multiprocessing.Pool(processes=args.no_thread)


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
                "segment_in" : os.path.join(args.seg_dir, segment_filename),
                "ang_in": os.path.join(args.ang_dir, ang_filename),
                "ang_out": os.path.join(args.out_dir, ang_filename),
                "coordinates" : {"x1" : 20, "y1": 20, "x2": 40, "y2": 40,
                        }
                }
            data.append(d)


    # execute multiprocessor script
    pool.map(parse_and_match, data)

    logger.info("Done!")

