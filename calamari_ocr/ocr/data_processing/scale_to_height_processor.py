import numpy as np
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor
from scipy.ndimage import interpolation


class ScaleToHeightProcessor(DataPreprocessor):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def _apply_single(self, data):
        return ScaleToHeightProcessor.scale_to_h(data, self.height)

    @staticmethod
    def scale_to_h(img, target_height, order=1, dtype=np.dtype('f'), cval=0):
        h, w = img.shape
        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        output = interpolation.affine_transform(
            1.0 * img,
            np.eye(2) / scale,
            order=order,
            output_shape=(target_height,target_width),
            mode='constant',
            cval=cval)

        output = np.array(output, dtype=dtype)
        return output
