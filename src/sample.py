import cv2

class Sample(object):
    def __init__(self, filename, crop_bottom=.10):
        self.img = cv2.imread(filename, 0)

        # crop the scale bar at the bottom
        new_height = int(self.img.shape[0] * (1-crop_bottom))
        self.img = self.img[:new_height, :]

    def get_image(self):
        return self.img
