from sample import Sample
import subprocess
import cv2

import numpy as np
import io

import skimage
import uuid

from skimage.transform import PolynomialTransform

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def compute_score(segment, grain):

    segmented_grain = grain >= 128
    segmented_segment = segment >= 128

    co_segmented = (segmented_grain & segmented_segment).sum()
    normalization = segmented_segment.sum() + segmented_grain.sum()

    score = 2 * co_segmented / normalization

    return score


# create a uuid for the current session to generate tmp png files. (nasty hack!)
tmp_uiid = uuid.uuid1()


# Create fitness function
def apply_distortion(segment, ebsd, points, polynom, segment_path_out=None, verbose=False):


    # turn the point into string
    str_buffer = io.BytesIO()
    np.savetxt(str_buffer, points, fmt='%i')

    # define in/out path
    in_segment = "/tmp/segment.align.{}.png".format(tmp_uiid)
    out_segment = "/tmp/segment.distord.{}.png".format(tmp_uiid)

    # save tmp image
    cv2.imwrite(in_segment, segment)

    if verbose:
        verbose_str = '-verbose'
    else:
        verbose_str = ''

    subprocess.run('convert {input_segment} {verbose} -virtual-pixel Black '
                   '-distort Polynomial "{polynom} {points}" '
                   '{output_distord}'
                   .format(input_segment=in_segment,
                           output_distord=out_segment,
                           points=str_buffer.getvalue(),
                           polynom=polynom,
                           verbose=verbose_str), shell=True)

    segment_distord = Sample(out_segment).get_image()

    # # Define the polynomial regression
    # model_i = Pipeline([('poly', PolynomialFeatures(degree=polynom, include_bias=True)),
    #                     ('linear', LinearRegression(fit_intercept=False, normalize=False))])
    #
    # model_j = Pipeline([('poly', PolynomialFeatures(degree=polynom, include_bias=True)),
    #                     ('linear', LinearRegression(fit_intercept=False, normalize=False))])
    #
    # def swap_cols(arr, frm, to):
    #     arr[:, [frm, to]] = arr[:, [to, frm]]
    #
    # sources, targets = points[:, :2], points[:, 2:]
    # swap_cols(sources, 1, 0)
    # swap_cols(targets, 1, 0)
    #
    # sources = sources.astype(np.float32)
    # sources[:, 0] = 2.*sources[:, 0] / segment.shape[0] - 1.
    # sources[:, 1] = 2.*sources[:, 1] / segment.shape[1] - 1.
    #
    # targets = targets.astype(np.float32)
    # targets[:, 0] = 2.*targets[:, 0] / segment.shape[0] - 1.
    # targets[:, 1] = 2.*targets[:, 1] / segment.shape[1] - 1.
    #
    # # Solve the regression system
    # model_i.fit(sources, targets[:, 0])
    # model_j.fit(sources, targets[:, 1])
    #
    #
    # # Define the image transformation
    # params = np.stack([model_i.named_steps['linear'].coef_, model_j.named_steps['linear'].coef_], axis=0)
    # transform = skimage.transform._geometric.PolynomialTransform(params)
    #
    # # Distord te image
    # segment_distord2 = skimage.transform.warp(segment, transform,
    #                                           cval=255 / 2,
    #                                           order=polynom,
    #                                           preserve_range=True)
    #
    # print("img_magick:  {0:.6f} \t sklearn: {1:.6f} ".format(
    #     compute_score(ebsd, segment_distord1),
    #     compute_score(ebsd, segment_distord2)))
    #
    #
    # # force segment to be either 0 or 255
    # # segment_distord[segment_distord < 128] = 0
    # # segment_distord[segment_distord >= 128] = 255

    if segment_path_out is not None:
        cv2.imwrite(segment_path_out, img=segment_distord)

    return segment_distord



class Aligner(object):

    def __init__(self, precrop=None, rescale=(1,1), normalization_score=None):
        self.precrop = precrop
        self.rescale = rescale
        self.normalization_score = normalization_score

    # Note that rotate should be done before the cropping to avoid moving the image center

    @staticmethod
    def __rotate__(segment, angle):

        if angle != 0:
            center_rotation = (segment.shape[0] / 2, segment.shape[1] / 2)
            M_rot = cv2.getRotationMatrix2D(center_rotation, angle, 1)
            segment = cv2.warpAffine(segment, M_rot, segment.shape[::-1])

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
    def __translate__(segment, tx, ty, shape):

        # Apply affine transformation
        M_aff = np.float32([[1, 0, tx], [0, 1, ty]])
        segment = cv2.warpAffine(segment, M_aff, shape)

        return segment

    @staticmethod
    def __postprocess__(segment, grain):

        score = compute_score(segment=segment, grain=grain)

        return segment, score

    def apply(self, segment, grain, tx, ty, angle):
        segment = self.__rotate__(segment=segment, angle=angle)
        segment = self.__crop__(segment=segment, precrop=self.precrop)
        segment = self.__rescale__(segment=segment, rescale=self.rescale)
        segment = self.__translate__(segment=segment, tx=tx, ty=ty, shape=grain.shape[::-1])

        segment, score = self.__postprocess__(segment, grain)

        return segment, score
