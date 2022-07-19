import serial, time
import threading
import cv2
from datetime import datetime 
import os
from PIL import Image
import numpy as np


def iniciarSensor(red):
    arduino = serial.Serial('COM3', 9600)
    while True:
        
        rawString = arduino.readline()
        if "True" in str(rawString):
            aux = True
            print("entro aqui 1")
            bandera, img_nombre = foto()
            if bandera:
                categoriasTest = os.listdir('dataset/test')
                im = 0
                im = Image.open(f"/fotos/{img_nombre}").resize((100,100))
                im = im.convert("RGB")
                im = np.asarray(im)
                im = np.array([im])
                predic = red.predict(im)
                print("Profesor: ", categoriasTest[np.argmax(predic)])
        
        print(rawString)

def inicio(modelo):
    t_start = threading.Thread(target=iniciarSensor, kwargs=(modelo,))
    t_start.start()


def foto():
    cap = cv2.VideoCapture(0)
    leido, frame = cap.read()
    nombre = ""
    bandera = False
    if leido == True:
        fecha = str(datetime.now())
        fecha = fecha.replace(".","-")
        fecha = fecha.replace(":","-")
        nombre = f"fotos/{fecha}.png"
        cv2.imwrite(f"fotos/{fecha}.png",frame)
        print("foto tomada correctamente")
        bandera = True
    cap.release()
    return bandera, nombre




