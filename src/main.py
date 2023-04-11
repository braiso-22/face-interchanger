import cv2 as cv
import dlib
import numpy as np

from image_utils import Img, Camera


def load_image(nombre) -> np.ndarray:
    route = "../images/"
    return Img.cargar_imagen(route + nombre)


def get_face_detector_and_predictor():
    detector = dlib.get_frontal_face_detector()
    # model from https://github.com/tzutalin/dlib-android/raw/master/data/shape_predictor_68_face_landmarks.dat
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
    cv.fillConvexPoly(mask, perimeter, 255)
    sniped = cv.bitwise_and(face, face, mask=mask)
    return sniped


def get_triangles_in_face(perimeter, landmarks):
    rect = cv.boundingRect(perimeter)
    subdiv = cv.Subdiv2D(rect)
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

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x, y) in points:
        cv.circle(img, (x, y), 1, (0, 0, 255), 2)
    for t in triangles_indexes:
        pt1 = points[t[0]]
        pt2 = points[t[1]]
        pt3 = points[t[2]]
        cv.line(img, pt1, pt2, (255, 0, 0), 1)
        cv.line(img, pt2, pt3, (255, 0, 0), 1)
        cv.line(img, pt1, pt3, (255, 0, 0), 1)
    return img


def mostrar_cara_pintada(img):
    detector, predictor = get_face_detector_and_predictor()

    img = img.copy()
    img_gray = Img.escala_grises(img)
    bordes_cara_rectangulo = detector(img)[0]

    landmarks = get_landmark_for_face(img_gray=img_gray, face=bordes_cara_rectangulo, predictor=predictor)
    points = np.array(landmarks, np.int32)
    landmarks_perimeter = cv.convexHull(points)
    # mask1 = np.zeros_like(cara1)
    # recorte = get_snip_face(landmarks_perimeter, cara1, mask1)
    triangles = get_triangles_in_face(perimeter=landmarks_perimeter, landmarks=landmarks)
    indexes_triangles = get_triangles_indexes(triangles, points)
    painted_image = pintar_detectado(img, bordes_cara_rectangulo, landmarks, indexes_triangles)
    painted_image = cv.cvtColor(painted_image, cv.COLOR_BGR2RGB)
    Img.mostrar(painted_image, True)


def get_triangle_points(triangle_indexes, points):
    point1, point2, point3 = points[triangle_indexes[0]], points[triangle_indexes[1]], points[triangle_indexes[2]]
    return point1, point2, point3


def get_rectangle_containing_triangle(triangle_indexes, points):
    p1, p2, p3 = get_triangle_points(triangle_indexes, points)
    triangle = np.array([p1, p2, p3], dtype=np.int32)
    rectangle = cv.boundingRect(triangle)
    return rectangle


def get_resized_points(triangle_index, points, x, y):
    p1, p2, p3 = get_triangle_points(triangle_index, points)
    return np.array(
        [
            [p1[0] - x, p1[1] - y],
            [p2[0] - x, p2[1] - y],
            [p3[0] - x, p3[1] - y],
        ],
        np.int32
    )


def pintar_lineas_triangulo(mascara, triangle_indexes, points):
    p1, p2, p3 = get_triangle_points(triangle_indexes, points)
    cv.line(mascara, p1, p2, 255)
    cv.line(mascara, p2, p3, 255)
    cv.line(mascara, p1, p3, 255)
    return mascara


def change_face(image1, image2):
    detector, predictor = get_face_detector_and_predictor()
    image1 = image1.copy()
    image2 = image2.copy()
    image2_gray = Img.escala_grises(image2)

    height, width, channels = image2.shape
    new_face = np.zeros((height, width, channels), np.uint8)

    cara_vacia1 = np.zeros_like(image1)
    cara_vacia2 = np.zeros_like(image2)

    bordes_cara1 = detector(image1)[0]
    bordes_cara2 = detector(image2)[0]

    landmarks1 = get_landmark_for_face(img_gray=Img.escala_grises(image1), face=bordes_cara1, predictor=predictor)
    landmarks2 = get_landmark_for_face(img_gray=Img.escala_grises(image2), face=bordes_cara2, predictor=predictor)

    points1 = np.array(landmarks1, np.int32)
    points2 = np.array(landmarks2, np.int32)

    landmarks_perimeter1 = cv.convexHull(points1)
    landmarks_perimeter2 = cv.convexHull(points2)

    triangles1 = get_triangles_in_face(perimeter=landmarks_perimeter1, landmarks=landmarks1)
    triangles2 = get_triangles_in_face(perimeter=landmarks_perimeter2, landmarks=landmarks2)

    triangle_indexes1 = get_triangles_indexes(triangles1, points1)
    triangle_indexes2 = get_triangles_indexes(triangles2, points2)
    result = None
    new_head_mask = None
    for triangle_index in triangle_indexes1:
        rectangle1 = get_rectangle_containing_triangle(triangle_index, points1)
        rectangle2 = get_rectangle_containing_triangle(triangle_index, points2)
        x1, y1, w1, h1 = rectangle1
        x2, y2, w2, h2 = rectangle2

        recorte1 = image1[y1:y1 + h1, x1:x1 + w1]
        mascara1 = np.zeros((h1, w1), np.uint8)
        mascara2 = np.zeros((h2, w2), np.uint8)

        resized_points1 = get_resized_points(triangle_index, points1, x1, y1)
        resized_points2 = get_resized_points(triangle_index, points2, x2, y2)

        cv.fillConvexPoly(mascara1, resized_points1, 255)
        cara_vacia1 = pintar_lineas_triangulo(cara_vacia1, triangle_index, points1)
        mascara1 = cv.bitwise_and(recorte1, recorte1, mask=mascara1)

        cv.fillConvexPoly(mascara2, resized_points2, 255)

        resized_points1 = np.float32(resized_points1)
        resized_points2 = np.float32(resized_points2)

        M = cv.getAffineTransform(resized_points1, resized_points2)
        warped_triangle = cv.warpAffine(recorte1, M, (w2, h2))
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mascara2)

        new_face_rectangle = new_face[y2:y2 + h2, x2:x2 + w2]
        new_face_rectangle_gray = cv.cvtColor(new_face_rectangle, cv.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv.threshold(new_face_rectangle_gray, 1, 255, cv.THRESH_BINARY_INV)
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        new_face_rectangle = cv.add(new_face_rectangle, warped_triangle)
        new_face[y2:y2 + h2, x2:x2 + w2] = new_face_rectangle

        new_face_mask = np.zeros_like(image2_gray)
        new_head_mask = cv.fillConvexPoly(new_face_mask, landmarks_perimeter2, 255)
        new_face_mask = cv.bitwise_not(new_head_mask)

        head_wihout_face = cv.bitwise_and(image2, image2, mask=new_face_mask)
        result = cv.add(head_wihout_face, new_face)

    (x, y, w, h) = cv.boundingRect(landmarks_perimeter2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv.seamlessClone(result, image2, new_head_mask, center_face2, cv.NORMAL_CLONE)
    swaped = cv.cvtColor(seamlessclone, cv.COLOR_BGR2RGB)

    return swaped


def operacion_frame(frame, params):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return change_face(params["face1"], frame), None


def main():
    cara1 = load_image("wismichu.jpg")
    cara2 = load_image("rajoy.jpg")

    mostrar_cara_pintada(cara2)
    mostrar_cara_pintada(cara1)
    changed = change_face(cara2, cara1)
    Img.mostrar(changed)

    Camera.video_capture(
        operacion=operacion_frame,
        operacion_params={
            "face1": cara1
        }
    )


if __name__ == '__main__':
    main()
