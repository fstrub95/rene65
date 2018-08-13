import collections
import numpy as np
Crop = collections.namedtuple("Crop", ["x_min", "x_max", "y_min", "y_max"])


# Note we recommend to define those values by with one manual operation
# It will


# Preprocessing
precrop = None
# precrop = Crop(
#     x_min=100, y_min=100,
#     x_max=1200, y_max=2800
# )




rescale_x = 0.4088
rescale_y = 0.4088

# Grid Search
range_angle = np.arange(-6, -4, 0.25)
range_translation_x = range(-90, -70, 1)
range_translation_y = range(-130, -100, 1)

