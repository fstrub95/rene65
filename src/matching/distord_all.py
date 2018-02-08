import subprocess
import re
import os
import copy
import numpy as np
import argparse

from matplotlib import pyplot as plt, cm

from sample import Sample
from crystallography.tsl import OimScan

if __name__ == "__main__":

    # Load conf
    parser = argparse.ArgumentParser('Calibrate them all!')

    parser.add_argument("-mesh_file", type=str, required=True, help="Path to the mesh file from align")
    parser.add_argument("-polynom", type=int, required=True, help="Polynom used to distord the image")

    parser.add_argument("-seg_dir", type=str, required=True, help="Path to the segmented image")
    parser.add_argument("-grain_dir", type=str, required=True, help="Path to the grain image")
    parser.add_argument("-ang_dir", type=str, required=True, help="Path to the ang file")
    parser.add_argument("-out_dir", type=str, required=True, help="Directory to store ang with phase")
    parser.add_argument("-tmp_dir", type=str, required=True, help="Directory to store intermediate results")

    args = parser.parse_args()

    # read the mesh-grid
    with open(args.mesh_file, "r") as file:
        mesh = file.read()

    print("Process all slices...")

    scores = [0]

    for segment_filename, grain_filename, ang_filename in zip(os.listdir(args.seg_dir), os.listdir(args.grain_dir), os.listdir(args.ang_dir)):

        id_segment = int(re.findall(r'\d+', segment_filename)[0])
        id_grain = int(re.findall(r'\d+', segment_filename)[0])
        id_ang = int(re.findall(r'R65_09_13_17_(\d+)_Mod.ang', ang_filename)[0])

        assert id_ang == id_grain, "Mismatch between file's id : {} vs {}".format(segment_filename, id_grain)
        assert id_ang == id_segment, "Mismatch between file's id : {} vs {}".format(segment_filename, ang_filename)

        # Distord the grain
        output_distord = '{tmp_dir}/segment_distord.buf.png'.format(tmp_dir=args.tmp_dir)
        subprocess.run('convert {input_segment} -virtual-pixel gray '
                       '-distort polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_segment=os.path.join(args.seg_dir, segment_filename),
                               output_distord=output_distord,
                               points=mesh,
                               polynom=args.polynom), shell=True)

        # Load distord grain
        segment = Sample(output_distord).get_image()
        grain = Sample(os.path.join(args.grain_dir, grain_filename)).get_image()

        # Compute and display score
        score = (grain == segment).sum() / (grain == 255).sum()
        print("Slice {0} : {1:.2f}%".format(id_segment, score*100))
        scores.append(score)

        # Create ang file
        ang_object = OimScan(os.path.join(args.ang_dir, ang_filename))

        # Create phase from segment
        phase = np.copy(segment)
        phase[np.where(segment > 255 / 2)] = 1
        phase[np.where(segment <= 255 / 2)] = 2
        ang_object.phase = phase

        # Match precipitate segmentation and EBSD
        phase1 = ang_object.phaseList[0]
        phase2 = copy.copy(ang_object.phaseList[0])
        phase2.number = 2
        phase2.formula = 'Ma'
        phase2.materialName = 'Matrix'
        ang_object.phaseList = [phase1, phase2]

        # Dump ang file
        ang_object.writeAng(os.path.join(args.out_dir, "R65_09_13_17_{}_Mod.phase.ang".format(str(id_segment).zfill(3))))

        # Dump overlap segment/iq
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(segment, interpolation='nearest', cmap=cm.gray)
        plt.imshow(ang_object.iq, interpolation='nearest', cmap=cm.jet, alpha=0.5)
        fig.savefig(os.path.join(args.tmp_dir, "overlap_seg_id.{}.png".format(id_segment)))

    print("----------------------------------------")
    print("No slices : {}".format(len(scores)))
    print("Score : {0:.2f} +/- {1:.2f}".format(np.mean(scores)*100, np.std(scores)*100))
    print("----------------------------------------")
