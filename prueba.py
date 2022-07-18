from PIL import Image
import numpy as np

from leerDataset import cargarDataSet
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import seaborn as sns
class Prueba:
    listaImagenes = ["dataset/test/scarlet/sj01.jpg","dataset/test/toreto/t01.jpeg","dataset/test/karla-scode/ke07.jpg","dataset/test/zac-efron/ze06.jpg"]

    def __init__(self,model,categorias) -> None:
        #self.inicarPrueba(model,categorias)
        self.imagenes_prueba(model)

    def inicarPrueba(self,model,categorias):
        for x in self.listaImagenes:
            im = 0
            im = Image.open(x).resize((100,100))
            im = im.convert("RGB")
            im = np.asarray(im)
            im = np.array([im])

            predic = model.predict(im)
           
            print(categorias[np.argmax(predic)])
        self.grafi(model)

    def imagenes_prueba(self, modelo):
        test_imagenes2 = []
        test_labels = []
        categoriasTest = os.listdir('dataset/test')
        l = 0
        for directorio in categoriasTest:
            aux = 1
            for imagen in os.listdir('dataset/test/'+directorio):
                img = Image.open('dataset/test/'+directorio+'/'+imagen).resize((100,100))
                img = np.asarray(img)
                if img.shape[2] == 4:
                    img = img[:,:,:3]
                test_imagenes2.append(img)
                img = np.array([img])
                if aux % 5 == 0:
                    predic = modelo.predict(img)
                    print(categoriasTest[np.argmax(predic)])
                test_labels.append(l)
                aux += 1
            l += 1

        test_imagenes2 = np.array(test_imagenes2)
        test_labels = np.array(test_labels)

        y_pred = np.argmax(modelo.predict(test_imagenes2), axis=1)
        y_true = test_labels

        self.graficar(y_pred, y_true)

    def graficar(self, y_predecida, y_verdadera):
        names = os.listdir('dataset/test')
        confusion_mtx = tf.math.confusion_matrix(y_verdadera, y_predecida)
        fig_matris = plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx,
                    xticklabels= names,
                    yticklabels= names,
                    annot=True,
                    fmt='g'
                )
        plt.title("Matriz de Confusion")
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.show()

    def grafi(self,modelo):
        names = ["scarlet","toreto","karla-scode","zac-efron"]
        imagenes,labels,categoria = cargarDataSet("test")
        audio = np.array(imagenes)
        labels = np.array(labels)
        y1 = np.argmax(modelo.predict(audio), axis=1)
        y2 = labels
        cfmtx = tf.math.confusion_matrix(y1,labels)
        fig = plt.figure(figsize=(10,8))
        sns.heatmap(
            cfmtx,
            xticklabels= names,
            yticklabels= names,
            annot=True,
            cmap='icefire',
            fmt='g'
        )
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.show()
        fig = plt.figure(figsize=(12, 7))
        print("llego aqui grafica matriz")