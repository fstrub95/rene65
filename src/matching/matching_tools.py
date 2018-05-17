from sample import Sample
import subprocess
import cv2

import numpy as np
import io

from skimage.transform import PolynomialTransform

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def compute_score(segment, ebsd, normalization=None):

    score = (segment == ebsd).sum() #score = equal(ebsd, segment).sum()

    if normalization is None:
        normalization = (ebsd == 255).sum()

    return score/normalization

import skimage
import uuid

# create a uuid for the current session to generate tmp png files. (nasty hack!)
tmp_uiid = uuid.uuid1()

# Create fitness function
def apply_distortion(ebsd, points, polynom, segment_path_out=None, use_image_magick=True,
                     verbose=False):

    if use_image_magick:

        # turn the point into string
        str_buffer = io.BytesIO()
        np.savetxt(str_buffer, points, fmt='%i')

        # define in/out path
        in_ebsd = "/tmp/ebsd.align.{}.png".format(tmp_uiid)
        out_ebsd = "/tmp/ebsd.distord.{}.png".format(tmp_uiid)

        # save tmp image
        cv2.imwrite(in_ebsd, ebsd)

        if verbose:
            verbose_str='-verbose'
        else:
            verbose_str = ''

        # execute imagemagick to perform the polynomial transformation
        subprocess.run('convert {input_ebsd} {verbose} -virtual-pixel gray '
                       '-distort polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_ebsd=in_ebsd,
                               output_distord=out_ebsd,
                               points=str_buffer.getvalue(),
                               polynom=polynom,
                               verbose=verbose_str), shell=True)

        ebsd_distord = Sample(out_ebsd).get_image()

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

        # Distord the image
        ebsd_distord = skimage.transform.warp(ebsd, transform, order=polynom, preserve_range=True)

    # force segment to be either 0 or 255
    ebsd_distord[ebsd_distord < 128] = 0
    ebsd_distord[ebsd_distord >= 128] = 255

    if segment_path_out is not None:
        cv2.imwrite(segment_path_out, img=ebsd_distord)

    return ebsd_distord


class Aligner(object):

    def __init__(self, precrop=None, rescale=(1,1), normalization_score=None):
        self.precrop = precrop
        self.rescale = rescale
        self.normalization_score = normalization_score

    # Note that rotate should be done before the cropping to avoid moving the image center

    @staticmethod
    def __rotate__(ebsd, angle):

        if angle != 0:
            center_rotation = (ebsd.shape[0] / 2, ebsd.shape[1] / 2)
            M_rot = cv2.getRotationMatrix2D(center_rotation, angle, 1)
            segment = cv2.warpAffine(ebsd, M_rot, ebsd.shape[::-1])

        return segment

    @staticmethod
    def __crop__(segment, precrop):

        # Note: we perform the crop after rotating!
        if precrop is not None:
            segment = segment[
                          precrop.x_min:precrop.x_max,
                          precrop.y_min:precrop.y_max]
        return segment

    @staticmethod
    def __rescale__(segment, rescale):
        if rescale[0] != 1 and rescale[1] != 1:
            segment = cv2.resize(segment, None,
                                 fx=rescale[0],
                                 fy=rescale[1],
                                 interpolation=cv2.INTER_AREA)
        return segment

    @staticmethod
    def __translate__(ebsd, tx, ty, shape):

        # Apply affine transformation
        M_aff = np.float32([[1, 0, tx], [0, 1, ty]])
        segment = cv2.warpAffine(ebsd, M_aff, shape)

        return segment

    @staticmethod
    def __postprocess__(segment, ebsd, normalization_score=None):

        segment[segment < 128] = 0
        segment[segment >= 128] = 255

        score = compute_score(segment=segment,
                              ebsd=ebsd,
                              normalization=normalization_score)

        return ebsd, score

    def apply(self, segment, ebsd, tx, ty, angle):
        ebsd = self.__rotate__(ebsd=ebsd, angle=angle)
        segment = self.__crop__(segment=segment, precrop=self.precrop)
        segment = self.__rescale__(segment=segment, rescale=self.rescale)
        ebsd = self.__translate__(ebsd=ebsd, tx=tx, ty=ty, shape=ebsd.shape[::-1])

        segment, score = self.__postprocess__(segment,
                                              ebsd,
                                              normalization_score=self.normalization_score)

        return ebsd, score











