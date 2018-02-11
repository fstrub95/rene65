import numpy as np

from multiprocessing.pool import ThreadPool
import os
import re

from sample import Sample
from matplotlib import pyplot as plt, cm
import argparse
from matching import matching_tools as mt

import cma
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])

def __main__(args=None):

    if args is None:

        parser = argparse.ArgumentParser('Image segmentation input!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-grain_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")
        parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

        parser.add_argument("-no_points", type=float, default=25, help="Ratio of image for eachpoint of the mesh")
        parser.add_argument("-std_pixels", type=float, default=7, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-max_sampling", type=int, default=1000, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-polynom", type=int, default=3, help="Order of the polynom to compute distorsion")

        parser.add_argument("-no_thread", type=int, default=2, help="Number of thread to run execute the segmentation")

        args = parser.parse_args()

    # check that grain and segment are the same slice
    id_segment = re.findall(r'\d+', os.path.basename(args.seg_ref_path))[0]
    id_grain = re.findall(r'\d+', os.path.basename(args.grain_ref_path))[0]
    id_slice = id_segment

    assert id_grain == id_segment, "Mismatch between file's id : {} vs {}".format(args.seg_ref_path, args.grain_ref_path)

    # Load segment (need to be preprocess by align.py)
    segment = Sample(args.seg_ref_path).get_image()

    # Load grain
    grain = Sample(args.grain_ref_path).get_image()
    grain = np.invert(grain)

    grain[grain < 128] = 1 # force non-grain values to have a different value than segment
    grain[grain >= 128] = 255

    # check that dimension match
    assert segment.shape == grain.shape, "Grain and shape must be of the same dimension"

    # compute initial score
    score_normalization = (grain == 255).sum()
    init_score = mt.compute_score(segment=segment, grain=grain, normalization=score_normalization)
    print("Init score : {0:.4f}".format(init_score))

    # Create initial mesh grid
    x = np.linspace(0, segment.shape[0], args.no_points)
    y = np.linspace(0, segment.shape[1], args.no_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    mesh = np.concatenate((xv, yv)).astype(np.int32)

    initial_points = mesh

    # prepare multithread distorsion
    pool = ThreadPool(processes=args.no_thread)

    # Create fitness function
    def fitness_fct(data):

        output_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=data.index)

        segment_distord = mt.apply_distortion(segment_in=args.seg_ref_path,
                                              segment_out=output_distord,
                                              polynom=args.polynom,
                                              points=data.points)

        score = mt.compute_score(segment=segment_distord, grain=grain, normalization=score_normalization)

        #Make the score negative as it is a minimization process
        score *= -1

        return score

    # prepare CME
    no_samples, best_score = 0, init_score
    es = cma.CMAEvolutionStrategy(initial_points, args.std_pixels)

    # Start black-box optimization
    while not es.stop() and no_samples < args.max_sampling:
        solutions = es.ask()

        #turn floating mesh into integer one (pixels are integer!)
        mesh_int = np.array(solutions, dtype=np.int32)

        # prepare data for multi-threading
        data = []
        for i, s in enumerate(mesh_int):
            xv_dist = s[:len(xv)]
            yv_dist = s[len(xv):]

            transformation = np.stack([xv, yv, xv_dist, yv_dist]).transpose()
            data.append(DataPoints(index=i, points=transformation))

        # compute the scores on separate threads (Not the imagemagick is then ran as a subprocess)
        scores = pool.map(fitness_fct, data)  # map keep the ordering

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
    out_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=id_slice)
    out_image = os.path.join(args.out_dir, "overlap.distord.slice{}.png".format(id_slice))
    out_points = os.path.join(args.out_dir, "mesh.{}.txt".format(id_slice))

    # Store points for transformation
    np.savetxt(out_points, transformation, fmt='%i')

    # Recompute best transformation
    final_segment = mt.apply_distortion(segment_in=args.seg_ref_path,
                        segment_out=out_distord,
                        polynom=args.polynom,
                        points=transformation)

    best_score = mt.compute_score(segment=final_segment, grain=grain, normalization=score_normalization)

    # Plot how grain/segment overlap
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(final_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)

    return best_score, final_segment, transformation

if __name__ == "__main__":
    __main__()








