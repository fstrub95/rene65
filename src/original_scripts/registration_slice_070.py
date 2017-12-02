
prefix = 'R65_09_13_17'
slice = 70
scan_name = '%s_%03d_Mod.ang' % (prefix, slice)
from crystallography.tsl import OimScan
scan = OimScan(scan_name)
print('OIM scan, spatial resolution is %g x %g' % (scan.xStep, scan.yStep))
iq = scan.iq  # size is 601x601
print(iq.shape)
pixel_size = 0.5  # micron

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt, cm
plt.imsave('%s_%03d_IQ.png' % (prefix, slice), iq, cmap=cm.gray, origin='upper')
plt.imshow(iq, interpolation='nearest', cmap=cm.gray, origin='upper')
plt.show()

import numpy as np
num_points = 6  # points per image
ref_points = np.zeros((num_points, 2))
tsr_points = np.zeros((num_points, 2))

# markers in Slice70.tif
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

from pymicro.view.vol_utils import compute_affine_transform

# compute the affine transform from the point set
translation, transformation = compute_affine_transform(ref_points, tsr_points)
invt = np.linalg.inv(transformation)
offset = -np.dot(invt, translation)
print(translation)
print(transformation)

invt
from pymicro.external.tifffile import TiffFile

sem = TiffFile('Slice%d.tif' % slice).asarray()[:2048,:].T  # imaged with scaling already applied
print('SEM shape', sem.shape)
compositeScan = OimScan.zeros_like(sem.T, resolution=(pixel_size, pixel_size))
compositeScan.sampleId = '%s_%03d' % (prefix, slice)

plt.imshow(sem.T, cmap=cm.gray)
plt.show()

from scipy import ndimage
# register ebsd
iq_reg = ndimage.interpolation.affine_transform(iq.T, invt, output_shape=sem.shape, offset=offset).T
eu0_reg = ndimage.interpolation.affine_transform(scan.euler[:, 0].T, invt, output_shape=sem.shape, offset=offset, order=0).T #bug in dimension
eu1_reg = ndimage.interpolation.affine_transform(scan.euler[:, 1].T, invt, output_shape=sem.shape, offset=offset, order=0).T
eu2_reg = ndimage.interpolation.affine_transform(scan.euler[:, 2].T, invt, output_shape=sem.shape, offset=offset, order=0).T
ci_reg = ndimage.interpolation.affine_transform(scan.ci.T, invt, output_shape=sem.shape, offset=offset, order=0, cval=-1).T
ph_reg = ndimage.interpolation.affine_transform(scan.phase.T, invt, output_shape=sem.shape, offset=offset, order=0).T
compositeScan.euler[:, :, 0] = eu0_reg
compositeScan.euler[:, :, 1] = eu1_reg
compositeScan.euler[:, :, 2] = eu2_reg
compositeScan.phase = ph_reg
compositeScan.ci = ci_reg
compositeScan.iq = iq_reg


fig = plt.figure(figsize=(15, 8))
plt.imshow(sem.T, interpolation='nearest', cmap=cm.gray)
plt.imshow(compositeScan.iq, interpolation='nearest', cmap=cm.jet, alpha=0.5)
plt.show()

plt.imsave('iq_reg_070.png', compositeScan.iq, cmap=cm.gray)
plt.imsave('sem_070.png', sem.T, cmap=cm.gray)

