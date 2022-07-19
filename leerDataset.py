import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def cargarDataSet(carpeta):
    x = 0
    imagenes = []
    labels   = []
    categoria = os.listdir("dataset/"+carpeta)
    for nombreCarpeta in categoria:
        for imagen in os.listdir("dataset/"+carpeta+"/"+nombreCarpeta):
            img = Image.open("dataset/"+carpeta+"/"+nombreCarpeta+"/"+imagen).resize((100,100))
            img = img.convert("RGB")
            img = np.asarray(img)
            imagenes.append(img)
            labels.append(x)
        x+=1 
    return (imagenes,labels,categoria)

def leer_dataset():

    categoriasTrain = os.listdir('dataset/train')

    print("Realizando aumento de datos")
    """ image_gen_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ) """

    image_gen_entrenamiento = ImageDataGenerator(rescale=1./255)

    data_gen_entrenamiento = image_gen_entrenamiento.flow_from_directory(batch_size=100,
        directory='dataset/Train',
        shuffle=True,
        target_size=(100,100),
        class_mode='binary'
    )

    total_entrenamiento = len(data_gen_entrenamiento.classes)

    image_gen_val = ImageDataGenerator(rescale=1./255)
    data_gen_validacion = image_gen_val.flow_from_directory(batch_size=100,
        directory='dataset/Validacion',
        target_size=(100, 100),
        class_mode='binary'
    )
    total_test = len(data_gen_validacion.classes)
    return data_gen_entrenamiento, data_gen_validacion, categoriasTrain, total_entrenamiento, total_test