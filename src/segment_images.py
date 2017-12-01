import argparse
import multiprocessing
import os
import logging
import numpy as np
import cv2
import math

from segmentation.evaluation import Evaluator
from segmentation.processing import LabyProcessing

from misc.logger import create_logger

from sample import Sample


# def show_img(img):
#     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('image', 600, 600)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#
# def save_img(img, filename="img"):
#     cv2.imwrite(os.path.join(folder_out, "{}.png".format(filename)), img)


def process_and_evaluate(data):

    filename_in = data["filename_in"]
    filename_out = data["filename_out"]
    evaluator = data["evaluator"]
    processor = data["processor"]

    logger = logging.getLogger()
    logger.info("Processing file: {}".format(filename_in))

    # Load image
    sample = Sample(filename_in, crop_bottom=0.10)

    # Compute segmentation
    final_img = processor.process(img=sample.get_image())

    # Compute error
    s, _ = evaluator.evaluate(final_img)

    # dump segmentation
    logger.info("Dumping file: {} - MAE: {}".format(filename_out, math.fabs(s)))
    cv2.imwrite(filename_out, final_img)

    return s


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-img_dir", type=str, required=True, help="Directory with input image")
    parser.add_argument("-seg_dir", type=str, required=True, help="Directory to output segmentation")
    parser.add_argument("-img_ext", type=str, default=".tif", help="Filter image in the directory according their extension")
    parser.add_argument("-no_thread", type=int, default=2, help="Number of thread to run execute the segmentation")

    args = parser.parse_args()

    # create technical python tools
    logger = create_logger(os.path.join(args.seg_dir, "log.txt"))
    pool = multiprocessing.Pool(processes=args.no_thread)


    # create functional python tools
    # TODO: use a config file
    evaluator = Evaluator(expected_ratio=0.116, classif_index=255)
    processor = LabyProcessing(114,
                               apply_denoising=True, denoising_factor=50,
                               apply_erode_dilate=True, no_erode=1,
                               apply_gaussblur=False)


    # Prepare sample for parallel computation
    data = []
    for sample_filename in os.listdir(args.img_dir):

        if sample_filename.endswith(args.img_ext):

            d = {
                "filename_in" : os.path.join(args.img_dir,sample_filename),
                "filename_out": os.path.join(args.seg_dir, sample_filename),
                "evaluator" : evaluator,
                "processor" : processor
            }
            data.append(d)


    # execute multiprocessor script
    diff = pool.map(process_and_evaluate, data)

    # Compute quadratic error (RMSE)
    error = np.sqrt(np.sum(np.square(diff))) / len(diff)

    # Print RMSE
    logger.info("RMSE : {}".format(error))




# Legacy code
# processor = MiLAProcessing(
#     gauss_kernel=random.choice([3,5]),
#     gauss_std=max(0.1, random.gauss(1, sigma=0.5)),
#     threshold_up = random.randint(130,220),
#     threshold_bottom = random.randint(20, 130),
#     self_kernel = random.randint(2,6),
#     use_erode= (random.randint(0,1) == 0)
# )
