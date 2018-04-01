from sample import Sample
import subprocess
import cv2

import numpy as np
import io

from skimage.transform import PolynomialTransform

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def compute_score(segment, grain, normalization=None):

    score = (grain == segment).sum()

    if normalization is None:
        normalization = (grain == 255).sum()

    return score/normalization

import skimage
# Create fitness function
def apply_distortion(segment, points, polynom, segment_path_out=None, use_image_magick=True):

    if use_image_magick:

        # turn the point into string
        str_buffer = io.BytesIO()
        np.savetxt(str_buffer, points, fmt='%i')

        # save tmp image
        cv2.imwrite("/tmp/segment.align.tmp.png", segment)

        # execute imagemagick to perform the polynomial transformation
        subprocess.run('convert  {input_segment} -virtual-pixel gray '
                       '-distort polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_segment="/tmp/segment.align.tmp.png",
                               output_distord="/tmp/segment.distord.tmp.png",
                               points=str_buffer.getvalue(),
                               polynom=polynom), shell=True)

        segment_distord = Sample("/tmp/segment.distord.tmp.png").get_image()

    else:

        # Define the polynomial regression
        model_x = Pipeline([('poly', PolynomialFeatures(degree=polynom)),
                          ('linear', LinearRegression(fit_intercept=False))])

        model_y = Pipeline([('poly', PolynomialFeatures(degree=polynom)),
                          ('linear', LinearRegression(fit_intercept=False))])

        # Solve the regression system
        model_x.fit(points[:, :2], points[:, 2])
        model_y.fit(points[:, :2], points[:, 3])

        # Define the image transformation
        params = np.stack([model_x.named_steps['linear'].coef_, model_y.named_steps['linear'].coef_], axis=0)
        transform = skimage.transform._geometric.PolynomialTransform(params)

        # Distord te image
        segment_distord = skimage.transform.warp(segment, transform, order=polynom, preserve_range=True)

    segment_distord[segment_distord < 128] = 0
    segment_distord[segment_distord >= 128] = 255

    # force segment to be either 0 or 255
    if segment_path_out is not None:
        cv2.imwrite(segment_path_out, img=segment_distord)

    return segment_distord


