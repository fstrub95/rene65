from sample import Sample
import subprocess
import cv2

import numpy as np
import io


def compute_score(segment, grain, normalization=None):

    score = (grain == segment).sum()

    if normalization is None:
        normalization = (grain == 255).sum()

    return  score/normalization


# Create fitness function
def apply_distortion(segment_in, segment_out, polynom, points, overwrite_segment=True):

    # turn the point into string
    str_buffer = io.BytesIO()
    np.savetxt(str_buffer, points, fmt='%i')

    # execute imagemagick to perform the polynomial transformation
    subprocess.run('convert {input_segment} -virtual-pixel gray '
                   '-distort polynomial "{polynom} {points}" '
                   '{output_distord}'
                   .format(input_segment=segment_in,
                           output_distord=segment_out,
                           points=str_buffer.getvalue(),
                           polynom=polynom), shell=True)

    segment_distord = Sample(segment_out).get_image()
    segment_distord[segment_distord < 128] = 0
    segment_distord[segment_distord >= 128] = 255

    # force segment to be either 0 or 255
    if overwrite_segment:
        cv2.imwrite(segment_out, img=segment_distord)

    return segment_distord


