#import tensorflow as tf
import cv2
import sensor_conector
import os
from leerDataset import cargarDataSet, leer_dataset

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from prueba import Prueba



def abrirCamera():
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        cv2.imshow("frame",frame)


        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    vid.release()
    cv2.destroyAllWindows()
    pass

def crearModelo2(img,validacion,categoria):
    total_entrenamiento = len(img.classes)
    total_test = len(validacion.classes)
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(categoria), activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    #img = np.asanyarray(img,dtype="float32")
    #labels = np.asarray(labels,dtype="float32")
    
    history = modelo.fit(
        img,
        steps_per_epoch=int(np.ceil(total_entrenamiento / float(100))),
        epochs = 20,
        validation_data= validacion,
        validation_steps=int(np.ceil(total_test / float(100)))
    )
    modelo.save("model/modeloprueba.h5")
    # Prueba(modelo,categoria)
    # plt.plot(history.history['loss'],label="loss")
    # plt.title("loss")
    # plt.show()

    # sensor_conector.inicio(modelo)
    # graficar(modelo)
    pass

if __name__ == "__main__":

    imagenes,validacion,categoria,total_entrenamiento, total_test = leer_dataset()
    if os.path.exists("model/modeloprueba.h5") == False:
        crearModelo2(imagenes,validacion,categoria)

    modelo = tf.keras.models.load_model("model/modeloprueba.h5") 
    Prueba(modelo,categoria)
    sensor_conector.inicio(modelo)
    # plt.plot(history.history['loss'],label="loss")
    # plt.title("loss")
    # plt.show()
    pass