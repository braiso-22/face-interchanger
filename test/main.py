import unittest

import dlib
import numpy as np

from src import main


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.detector, _ = main.get_face_detector_and_predictor()
        self.face1 = main.load_image("wismichu.jpg")

    def test_load_image_returns_ndarray(self):
        image = type(main.load_image("rajoy.jpg"))
        self.assertTrue(image is np.ndarray)

    def test_load_image_returns_none_without_str(self):
        image = main.load_image("")
        self.assertTrue(image is None)

    def test_get_face_detector_returns_detector_and_predictor(self):
        detector, predictor = main.get_face_detector_and_predictor()
        self.assertTrue(type(detector) is dlib.fhog_object_detector)
        self.assertTrue(type(predictor) is dlib.shape_predictor)


if __name__ == '__main__':
    unittest.main()
