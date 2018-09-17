import numpy as np

import os

from distutils.util import strtobool
from sample import Sample
from matplotlib import pyplot as plt, cm
import argparse
from matching import matching_tools as mt

from ang.write_core_ang import Ang
from ang.phase import Phase

import cma
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])


def __main__(args=None):
    if args is None:
        parser = argparse.ArgumentParser('Image segmentation input!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-ebsd_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-ang_ref_path", type=str, required=True, help="Path to the ang file")

        parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")

        parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

        parser.add_argument("-no_points", type=float, default=None, help="Ratio of image for eachpoint of the mesh")
        parser.add_argument("-no_points_step", type=float, default=50, help="Use an absolute step size instead of no_points")
        parser.add_argument("-std_pixels", type=float, default=5, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-max_sampling", type=int, default=2500, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-polynom", type=int, default=3, help="Order of the polynom to compute distorsion")

        parser.add_argument("-invert_ebsd", type=lambda x: bool(strtobool(x)), default="False", help="Put True if background is white")
        parser.add_argument("-image_magic", type=lambda x: bool(strtobool(x)), default="False", help="use image magick")

        args = parser.parse_args()

    # check that grain and segment are the same slice
    id_slice = 0

    # Load segment (need to be preprocess by align.py)
    segment_align = Sample(args.seg_ref_path).get_image()

    # Load grain
    ebsd = Sample(args.ebsd_ref_path).get_image()
    if args.invert_ebsd:
        ebsd = np.invert(ebsd)  # we need the background to be black (default color for numpy transformation)

    # check that dimension match
    assert segment_align.shape == ebsd.shape, "Grain and shape must be of the same dimension"

    # compute initial score
    init_score = mt.compute_score(segment=segment_align, ebsd=ebsd)
    print("Init score : {0:.4f}".format(init_score))

    # Create initial mesh grid
    if args.no_points_step is not None:
        x = np.arange(0, segment_align.shape[1], args.no_points_step)
        y = np.arange(0, segment_align.shape[0], args.no_points_step)
    else:
        x = np.linspace(0, segment_align.shape[1], args.no_points)
        y = np.linspace(0, segment_align.shape[0], args.no_points)

    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    initial_points = np.concatenate((xv, yv)).astype(np.int32)

    # prepare CME
    no_samples, best_score = 0, init_score
    es = cma.CMAEvolutionStrategy(initial_points, args.std_pixels)  # , {'seed': 123})

    all_scores = []

    # Start black-box optimization
    while not es.stop() and no_samples < args.max_sampling:
        solutions = es.ask()

        # turn floating mesh into integer one (pixels are integer!)
        mesh_int = np.array(solutions, dtype=np.int32)

        # prepare data for multi-threading
        scores = []

        i = 0
        for s in mesh_int:
            xv_dist = s[:len(xv)]
            yv_dist = s[len(xv):]

            transformation = np.stack([xv, yv, xv_dist, yv_dist]).transpose()

            ebsd_distord, _ = mt.apply_distortion_ebsd(segment=segment_align, ebsd=ebsd,
                                                    points=transformation,
                                                    polynom=args.polynom,
                                                    use_img_magic=args.image_magic)

            score = mt.compute_score(segment=segment_align, ebsd=ebsd_distord)
            all_scores.append(score)

            # Make the score negative as it is a minimization process
            score *= -1
            scores.append(score)

        # store no_samples
        no_samples += len(scores)

        es.tell(solutions, scores)
        es.disp()

    # Display results
    es.result_pretty()

    # retrieve the best mesh
    final_mesh = es.result.xbest

    xv_dist = final_mesh[:len(xv)]
    yv_dist = final_mesh[len(xv):]
    transformation = np.stack([xv, yv, xv_dist, yv_dist]).transpose()
    transformation = transformation.astype(np.int32)

    # Define path to save results
    out_distord = '{tmp_dir}/ebsd_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=id_slice)
    out_image = os.path.join(args.out_dir, "overlap.distord_ebsd.slice{}.png".format(id_slice))
    out_points = os.path.join(args.out_dir, "mesh.{}.txt".format(id_slice))

    # Store points for transformation
    np.savetxt(out_points, transformation, fmt='%i')

    # Recompute best transformation
    final_ebsd, params = mt.apply_distortion_ebsd(segment=segment_align, ebsd=ebsd,
                                                  ebsd_path_out=out_distord,
                                                  polynom=args.polynom,
                                                  points=transformation,
                                                  verbose=True)

    best_score = mt.compute_score(segment=segment_align,
                                  ebsd=final_ebsd)

    print("Final best score : {0:.4f}".format(best_score))

    # Plot how grain/segment overlap
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(segment_align, interpolation='nearest', cmap=cm.gray)
    plt.imshow(final_ebsd, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)

    # fig, ax = plt.subplots(figsize=(15, 8))
    # ax.imshow(final_segment)
    # ax.plot(transformation[:, 2], transformation[:, 3], '.')
    # plt.show()

    # Create ang file
    ang = Ang(args.ang_ref_path)
    ang.warp(params)

    # TODO(fstrub): Allow user to define it
    phase = Phase.create_from_phase(id=2, name='Prec', formula='NiAl', phase=ang.phases[0])
    ang.add_phase(segment_align=segment_align, phase=phase)
    ang.dump_file(os.path.join(args.out_dir, os.path.basename(args.ang_ref_path)))

    return best_score, final_ebsd, transformation, all_scores


if __name__ == "__main__":
    __main__()
