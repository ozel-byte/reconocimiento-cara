#import tensorflow as tf
import cv2
import sensor_conector
import os
from leerDataset import cargarDataSet, leer_dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from  keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

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
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(100, 100, 3)),
        # tf.keras.layers.MaxPooling2D(2, 2),

        # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),

        # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(len(categoria), activation='softmax')
        tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(100, 100,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(len(categoria), activation="softmax")
    ])

    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    #img = np.asanyarray(img,dtype="float32")
    #labels = np.asarray(labels,dtype="float32")
    # model = Sequential()
    # model.add(Conv2D(32,(3,3),input_shape=(100,100,3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dense(256, kernel_constraint=maxnorm(3)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
        
    # model.add(Dense(128, kernel_constraint=maxnorm(3)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
  
    # model.add(Activation('softmax'))

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # print(model.summary())
    history = modelo.fit(
        img,
        steps_per_epoch=int(np.ceil(total_entrenamiento / float(100))),
        epochs = 50,
        validation_data= validacion,
        validation_steps=int(np.ceil(total_test / float(100)))
    )
    # scores=model.evaluate(img)
    # print(scores[1]*100)
    modelo.save("model/modeloprueba.h5")
    # Prueba(modelo,categoria)
    # plt.plot(history.history['loss'],label="loss")
    # plt.title("loss")
    # plt.show()

    # sensor_conector.inicio(modelo)
    # graficar(modelo)
    pass


def crearModelo3(imagenes,labels,categoria):
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(100,100,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
        
    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
  
    model.add(Activation('softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # img = np.asanyarray(img,dtype="float32")
    # labels = np.asarray(labels,dtype="float32")
    print(model.summary())
    imagenes = np.asanyarray(imagenes,dtype="float32")
    labels = np.asarray(labels,dtype="float32")
    history = model.fit(
        imagenes,
        labels,
        epochs = 20,
    )
    model.save("model/modeloprueba.h5")
    pass

if __name__ == "__main__":

    imagenes,validacion,categoria,total_entrenamiento, total_test = leer_dataset()
    #imagenes,labels,categoria = cargarDataSet("train")
    if os.path.exists("model/modeloprueba.h5") == False:
        crearModelo2(imagenes,validacion,categoria)
        #crearModelo3(imagenes,labels,categoria)

    modelo = tf.keras.models.load_model("model/modeloprueba.h5") 
    Prueba(modelo,categoria)
    sensor_conector.inicio(modelo)
    # plt.plot(history.history['loss'],label="loss")
    # plt.title("loss")
    # plt.show()
    pass