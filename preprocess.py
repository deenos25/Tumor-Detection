import numpy as np
import cv2

class TumorImageProcessing:

    def __init__(self, image, box_coors=None, class_labels=None, max_tumors=1):
        self.image = image
        self.box_coors = box_coors
        self.class_labels = class_labels
        self.max_tumors = max_tumors

    def resize_and_normalize(self):
        image_resized = cv2.resize(self.image, (512, 512))
        image_resized_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
        return image_resized_normalized

    def create_output_array(self):
        output_arr = np.zeros((self.max_tumors, 6), dtype=object)
        for i in range(len(self.box_coors)):
            output_arr[i] = np.array([1] + list(self.box_coors[i]) + [self.class_labels[i]], dtype=object)
        return output_arr







