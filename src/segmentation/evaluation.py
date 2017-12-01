import numpy as np

class Evaluator(object):

    def __init__(self, expected_ratio, classif_index=255):
        self.classif_index = classif_index
        self.expected_ratio = expected_ratio

    def evaluate(self, img_to_eval):
        pixel_segmented = img_to_eval.size - np.count_nonzero(img_to_eval)
        pixel_total = img_to_eval.size

        predicted_ratio = 1.0*pixel_segmented / pixel_total

        score = predicted_ratio - self.expected_ratio

        return score, predicted_ratio


