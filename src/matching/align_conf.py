import collections
import numpy as np
Crop = collections.namedtuple("Crop", ["x_min", "x_max", "y_min", "y_max"])


# Note we recommend to define those values by with one manual operation
# It will


# Preprocessing
precrop = Crop(
    x_min=0, y_min=700,
    x_max=1565, y_max=2715
)

rescale_x = 0.3265
rescale_y = 0.3265

# Grid Search
range_angle = np.arange(-2.0, 2.0, 0.5)
range_translation_x = range(-25, 5, 1)
range_translation_y = range(105, 135, 1)

