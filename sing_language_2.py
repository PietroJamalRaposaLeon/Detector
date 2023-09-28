import cv2
import mediapipe as mp


camera = cv2.VideoCapture(0)


desenho = mp.solutions.drawing_utils
maos = mp.solutions.hands


detectar_maos = maos.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)


def desenharMaos(image, marcas_maos):
    if marcas_maos:
        for marcas in marcas_maos:
            desenho.draw_landmarks(image, marcas, maos.HAND_CONNECTIONS)


while True:
    success, image = camera.read()


    image = cv2.flip(image, 1)


    resultado = detectar_maos.process(image)


    marcas_maos = resultado.multi_hand_landmarks


    desenharMaos(image, marcas_maos)


    cv2.imshow("Webcam", image)


    key = cv2.waitKey(1)


    if key == 32:
        break


cv2.destroyAllWindows()