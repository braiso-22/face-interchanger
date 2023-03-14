import unittest

import numpy as np

from src import main


class MyTestCase(unittest.TestCase):

    def test_load_image_returns_ndarray(self):
        image = type(main.load_image("rajoy.jpg"))
        self.assertTrue(image is np.ndarray)

    def test_load_image_returns_none_without_str(self):
        image = main.load_image("")
        self.assertTrue(image is None)


if __name__ == '__main__':
    unittest.main()
