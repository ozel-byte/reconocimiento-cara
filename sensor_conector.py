from cgi import test
from pyexpat import model
from tabnanny import verbose
import serial, time
import threading
import cv2
from datetime import datetime 
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def iniciarSensor(red):
    arduino = serial.Serial('COM5', 9600)
    while True:
        
        rawString = arduino.readline()
        if "True" in str(rawString):
            aux = True
            print("entro aqui 1")
            bandera, img_nombre = foto()
            if bandera:
                categoriasTest = os.listdir('dataset/test')

                img = tf.keras.utils.load_img(img_nombre, target_size = (100, 100))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                # prediccion = red.predict(img_array)
                # score = tf.nn.softmax(prediccion)
                print(img_nombre)
                print(red.summary())
                scores = red.evaluate(img_array)
                print(scores[1]*100)
                #plt.imshow(img)
                #plt.show()

                #im = Image.open(f"{img_nombre}").resize((100,100))
                #im = im.convert("RGB")
                #im = np.asarray(im)
                #predic = red.predict(im)
                #test_loss, test_acc = red.evaluate(im,categoriasTest,verbose=2)
                #print(test_acc)
                #print(predic[0])
                #pre = np.argmax(predic[0])
                #print(pre)
                #print("Esta imagen probablemente pertenece a: {} con un {:.2f} porcentaje de seguridad".format(categoriasTest[np.argmax(score)], 100*np.max(score)))

                """if 100 * np.max(predic) < 60.0:
                    #print("Esta imagen probablemente pertenece a: {} con un {:.2f} porcentaje de seguridad".format(categoriasTest[np.argmax(score)], 100*np.max(score)))
                    print("No se reconoce a este persona verifique con su superior")
                else:
                    print("Esta imagen probablemente pertenece a: {} con un {:.2f} porcentaje de seguridad".format(categoriasTest[np.argmax(score)], 100*np.max(score)))"""

                #print("Profesor: ", categoriasTest[np.argmax(predic)])
        
        print(rawString)

def inicio(modelo):
    t_start = threading.Thread(target=iniciarSensor, args=(modelo,))
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




