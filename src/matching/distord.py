import numpy as np
import io
import subprocess
import multiprocessing
import os

from sample import Sample
from matplotlib import pyplot as plt, cm

import argparse

import cma
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Image segmentation input!')

    parser.add_argument("-seg_path", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_path", type=str, required=True, help="Path to the grain image")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")
    parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

    parser.add_argument("-step_ratio", type=float, default=0.125, help="Ratio of image for eachpoint of the mesh")
    parser.add_argument("-std_ratio", type=float, default=0.01, help="How far are going to look around the mesh ground (% of the image dimension)")
    parser.add_argument("-max_sampling", type=int, default=20000, help="How far are going to look around the mesh ground (% of the image dimension)")
    parser.add_argument("-polynom", type=int, default=3, help="Order of the polynom to compute distorsion")

    parser.add_argument("-no_thread", type=int, default=2, help="Number of thread to run execute the segmentation")

    args = parser.parse_args()

    segment = Sample(args.seg_path).get_image()
    grain = Sample(args.grain_path).get_image()

    image_id = 0  # TODO  ; parse image to retrieve id

    assert segment.shape == grain.shape, "Grain and shape must be of the same dimension"

    # Create mesh grid
    x = np.linspace(0, segment.shape[0], segment.shape[0]*args.step_ratio)
    y = np.linspace(0, segment.shape[1], segment.shape[1]*args.step_ratio)
    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    mesh = np.concatenate((xv, yv))

    # prepare multithread distorsion
    pool = multiprocessing.Pool(processes=args.no_thread)

    # Create fitness function
    def fitness_fct(data):

        str_buffer = io.BytesIO()
        np.savetxt(str_buffer, data.points, fmt='%i')

        output_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=data.index)

        subprocess.run('convert {input_segment} -virtual-pixel gray '
                       '-distort polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_segment=args.seg_path,
                               output_distord=output_distord,
                               points=str_buffer.getvalue(),
                               polynom=args.polynom), shell=True)

        segment_distord = Sample(output_distord).get_image()

        segment_distord[segment_distord < 128] = 0
        segment_distord[segment_distord >= 128] = 255

        score = -(grain == segment_distord).sum()

        return score

    # prepare CME
    no_samples, best_score = 0, float("inf")
    best_solution = None
    es = cma.CMAEvolutionStrategy(mesh, args.pixel_std)

    # Start black-box optimization
    while not es.stop() or no_samples > args.max_sampling:
        solutions = es.ask()

        solutions_int = np.array(solutions, dtype=np.int32)

        # prepare data for multi-threading
        data = []
        for i, s in enumerate(solutions_int):
            xv_dist = s[:len(xv)]
            yv_dist = s[len(xv):]

            transformation = np.array([xv, yv, xv_dist, yv_dist]).transpose()
            data.append(DataPoints(index=i, points=transformation))

        scores = pool.map(fitness_fct, data)  # map keep the ordering

        # store intermediate best sample
        cur_best_score = min(scores)
        if cur_best_score < best_score:
            best_solution = solutions_int[scores.index(cur_best_score)]

        no_samples += len(scores)

        es.tell(solutions, scores)
        es.disp()

    # Display results
    es.result_pretty()

    # Define path to save results
    out_distord = '{tmp_dir}/segment_distord.{index}.png'.format(tmp_dir=args.tmp_dir, index=image_id)
    out_image = os.path.join(args.out_dir, "overlap.slice{}.png".format(image_id))
    out_points = os.path.join(args.out_dir, "points.slice{}.txt".format(image_id))

    # Store points for transformation
    np.savetxt(out_points, best_solution, fmt='%i')

    # Recompute best transformation
    best_data = DataPoints(index=image_id, points=best_solution)
    fitness_fct(best_data)

    # Plot how grain/segment overlap
    distord_segment = Sample(out_distord).get_image()

    fig = plt.figure(figsize=(15, 8))
    plt.imshow(distord_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(grain, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)










