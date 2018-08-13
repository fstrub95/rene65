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
range_angle = np.arange(-3.0, -1, 0.25)
range_translation_x = range(-30, 10, 1)
range_translation_y = range(100, 140, 1)

