import cv2
import dlib
import numpy as np

from image_utils import Img


def load_image(nombre) -> np.ndarray:
    route = "../images/"
    return Img.cargar_imagen(route + nombre)


def get_face_detector_and_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def get_landmark_for_face(img_gray, face, predictor):
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for point in range(0, 68):
        x = landmarks.part(point).x
        y = landmarks.part(point).y
        landmarks_points.append((x, y))
    return landmarks_points


def get_snip_face(perimeter, face, mask):
    cv2.fillConvexPoly(mask, perimeter, 255)
    sniped = cv2.bitwise_and(face, face, mask=mask)
    return sniped


def get_triangles_in_face(perimeter, landmarks):
    rect = cv2.boundingRect(perimeter)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return triangles


def get_triangle_indexes(triangle, landmark_points):
    point1 = (triangle[0], triangle[1])
    point2 = (triangle[2], triangle[3])
    point3 = (triangle[4], triangle[5])

    index_pt1 = np.where((landmark_points == point1).all(axis=1))
    index_pt1 = index_pt1[0][0]

    index_pt2 = np.where((landmark_points == point2).all(axis=1))
    index_pt2 = index_pt2[0][0]

    index_pt3 = np.where((landmark_points == point3).all(axis=1))
    index_pt3 = index_pt3[0][0]

    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        return triangle
    return None


def get_triangles_indexes(triangles, points):
    indexes_triangles = []
    for triangle in triangles:
        indexes = get_triangle_indexes(triangle, points)
        if indexes is not None:
            indexes_triangles.append(indexes)
    return indexes_triangles


def pintar_detectado(img, rectangulo, points, triangles_indexes):
    img = img.copy()
    x1 = rectangulo.left()
    y1 = rectangulo.top()
    x2 = rectangulo.right()
    y2 = rectangulo.bottom()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x, y) in points:
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
    for t in triangles_indexes:
        pt1 = points[t[0]]
        pt2 = points[t[1]]
        pt3 = points[t[2]]
        cv2.line(img, pt1, pt2, (255, 0, 0), 1)
        cv2.line(img, pt2, pt3, (255, 0, 0), 1)
        cv2.line(img, pt1, pt3, (255, 0, 0), 1)
    return img


def mostrar_cara_pintada(img, detector, predictor):
    img = img.copy()
    img_gray = Img.escala_grises(img)
    bordes_cara_rectangulo = detector(img)[0]
    landmarks1 = get_landmark_for_face(img_gray=img_gray, face=bordes_cara_rectangulo, predictor=predictor)
    points = np.array(landmarks1, np.int32)
    landmarks_perimeter = cv2.convexHull(points)
    # mask1 = np.zeros_like(cara1)
    # recorte = get_snip_face(landmarks_perimeter, cara1, mask1)
    triangles = get_triangles_in_face(perimeter=landmarks_perimeter, landmarks=landmarks1)
    indexes_triangles = get_triangles_indexes(triangles, points)
    painted_image = pintar_detectado(img, bordes_cara_rectangulo, landmarks1, indexes_triangles)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB)
    Img.mostrar(painted_image, True)


def main():
    cara1 = load_image("rajoy.jpg")
    cara2 = load_image("wismichu.jpg")

    detector, predictor = get_face_detector_and_predictor()

    mostrar_cara_pintada(cara1, detector, predictor)
    mostrar_cara_pintada(cara2, detector, predictor)
    pass


if __name__ == '__main__':
    main()
