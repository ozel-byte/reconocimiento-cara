import cv2
from cv2 import imwrite



def foto():
    cap = cv2.VideoCapture(0)
    leido, frame = cap.read()
    if leido == True:
        cv2,imwrite("fotos/foto.png",frame)
        print("foto tomada correctamente")
    else:
        print("no se tomo la foto")
    cap.release()


foto()