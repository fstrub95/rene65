import numpy as np
import json

num_points = 6  # points per image
ref_points = np.zeros((num_points, 2))
tsr_points = np.zeros((num_points, 2))

ref_points[0] = [1383, 705]
ref_points[1] = [1371, 783]
ref_points[2] = [1881, 879]
ref_points[3] = [1197, 882]
ref_points[4] = [1155, 870]
ref_points[5] = [1122, 867]

# markers in image R65_09_13_17_070_Mod.ang (IQ)
tsr_points[0] = [353, 588]
tsr_points[1] = [350, 634]
tsr_points[2] = [641, 702]
tsr_points[3] = [252, 685]
tsr_points[4] = [229, 677]
tsr_points[5] = [209, 675]



num_slices = 325

idslice_2_points = {}
for id_slice in range(num_slices):

    if id_slice < 20:

        slice = {
            "ref_points": ref_points.tolist(),
            "tsr_points": tsr_points.tolist(),
        }

    idslice_2_points[id_slice] = slice

with open('calibration_points.json', "w") as file:
    json.dump(idslice_2_points, file)

print("Done!")

# with open('C:\Users\SteveJobs\PycharmProjects\rene65', "r") as file:
#     data = json.load(file)