import numpy as np

from image_utils import Img


def load_image(nombre) -> np.ndarray:
    route = "../images/"
    return Img.cargar_imagen(route + nombre, mostrar=True)


def main():
    load_image("rajoy.jpg")
    pass


if __name__ == '__main__':
    main()
