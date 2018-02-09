import numpy as np
import io
import subprocess
from multiprocessing.pool import ThreadPool
import os
import re

from sample import Sample
from matplotlib import pyplot as plt, cm

import argparse

import cma
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_ref_path", type=str, required=True, help="Path to the grain image")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")
    parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

    parser.add_argument("-no_points", type=float, default=30, help="Ratio of image for eachpoint of the mesh")
    parser.add_argument("-std_pixels", type=float, default=3, help="How far are going to look around the mesh ground (% of the image dimension)")
    parser.add_argument("-max_sampling", type=int, default=3000, help="How far are going to look around the mesh ground (% of the image dimension)")
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

    grain[grain < 128] = 1
    grain[grain >= 128] = 255

    # check that dimension match
    assert segment.shape == grain.shape, "Grain and shape must be of the same dimension"

    # compute initial score
    score_normalization = (grain == 255).sum()
    init_score = (grain == segment).sum() / score_normalization
    print("Init score : {0:.4f}".format(init_score))

    # Create initial mesh grid
    x = np.linspace(0, segment.shape[0], args.no_points)
    y = np.linspace(0, segment.shape[1], args.no_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    mesh = np.concatenate((xv, yv)).astype(np.int32)

    # define additional affine transformation TODO
    # affine transformation tx, ty, angle
    # affine_transformation = [0, 0, 0]

    initial_points = mesh # np.concatenate([mesh, affine_transformation])

    # prepare multithread distorsion
    pool = ThreadPool(processes=args.no_thread)

    # Create fitness function
    def fitness_fct(data):

        str_buffer = io.BytesIO()
        np.savetxt(str_buffer, data.points, fmt='%i')

        output_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=data.index)

        subprocess.run('convert {input_segment} -virtual-pixel gray '
                       '-distort polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_segment=args.seg_ref_path,
                               output_distord=output_distord,
                               points=str_buffer.getvalue(),
                               polynom=args.polynom), shell=True)

        segment_distord = Sample(output_distord).get_image()

        segment_distord[segment_distord < 128] = 0
        segment_distord[segment_distord >= 128] = 255

        score = (grain == segment_distord).sum() / score_normalization
        score *= -1

        return score

    # prepare CME
    no_samples, best_score = 0, init_score
    best_solution = None
    es = cma.CMAEvolutionStrategy(initial_points, args.std_pixels)

    # Start black-box optimization
    while not es.stop() and no_samples < args.max_sampling:
        solutions = es.ask()

        #[tx, ty, angle] = solutions[-len(affine_transformation):]
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

        # store intermediate best sample
        cur_best_score = min(scores)
        if cur_best_score < best_score:
            best_score = cur_best_score
            best_solution = mesh_int[scores.index(cur_best_score)]

        no_samples += len(scores)

        es.tell(solutions, scores)
        es.disp()

    # Display results
    es.result_pretty()

    # retrive the best mesh
    final_mesh = es.result.xbest
    final_mesh = np.array(final_mesh, dtype=np.int32)

    xv_dist = final_mesh[:len(xv)]
    yv_dist = final_mesh[len(xv):]

    transformation = np.stack([xv, yv, xv_dist, yv_dist]).transpose()

    # Define path to save results
    out_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=id_slice)
    out_image = os.path.join(args.out_dir, "overlap.distord.slice{}.png".format(id_slice))
    out_points = os.path.join(args.out_dir, "points.slice{}.txt".format(id_slice))

    # Store points for transformation
    np.savetxt(out_points, transformation, fmt='%i')

    # Recompute best transformation
    best_data = DataPoints(index=id_slice, points=transformation)
    fitness_fct(best_data)

    # Plot how grain/segment overlap
    distord_segment = Sample(out_distord).get_image()
    distord_segment[distord_segment  < 128] = 0
    distord_segment[distord_segment >= 128] = 255

    fig = plt.figure(figsize=(15, 8))
    plt.imshow(distord_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)










