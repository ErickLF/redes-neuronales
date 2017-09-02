# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:40:07 2017

@author: Belen Chavarría
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

os.chdir('C:/Users/Erick/Documents/2017-1/Reconocimiento_de_Patrones')
datos = pd.read_csv("AirQualityClaseKMeans.csv")

#datos = pd.read_csv("voice_limpia_4_L.csv")

datos = shuffle(datos.iloc[:])

datosX = datos.iloc[:,0: 20].values
datosY = datos.iloc[:, 20].values

datos_muestra = shuffle(datos.iloc[0:100])
# Etiqueta de clase de cada vector ejemplo
Y = datos_muestra.iloc[0:100, 20].values
# Vector de características
X = datos_muestra.iloc[0:100, 0:20].values #las 100 primeras filas de todas las columnas sin label
# Tasa de aprendizaje
eta=0.01
# Vector de pesos inicial
w = np.zeros(X.shape[1]) #vector de 20 ceros
# Datos para entrenameinto y prueba
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.5)
# Entrenamiento
#print("Vector\tTarget\toutput\terror\tw")
vector = 1
for xi, target in zip(train_x, train_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    #if(error != 0):
        #print(vector, "\t", target, "\t", output, "\t", error, "\t", w)
    vector += 1
    
# Prueba
errores = 0
for xi, target in zip(train_x, train_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores de entrenamiento mal clasificados de {} ({}%)".format(errores, len(train_x), 
                                                        errores/len(train_x)*100))

errores = 0
for xi, target in zip(test_x, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores de prueba mal clasificados de {} ({}%)".format(errores, len(test_x), 
                                                        errores/len(test_x)*100))

print("Segunda ronda")
errores=0
train2 = shuffle(list(zip(train_x, train_y)))
for xi, target in train2:
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    if (target != output) :
        errores += 1
print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
      format(errores, len(train_y), errores/len(train_y)*100))

# Prueba
errores = 0
for xi, target in zip(test_x, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores de prueba mal clasificados de {} ({}%)".format(errores, len(test_y), 
                                                        errores/len(test_y)*100))

print("Tercera ronda con todos los datos...")
vectores = 0
for xi, target in zip(datosX, datosY):
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    if (target != output) :
        vectores += 1
print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
      format(vectores, len(datosY), vectores/len(datosY)*100))
    
# Prueba
errores = 0
for xi, target in zip(test_x, test_y):
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_y), 
                                                        errores/len(test_y)*100))


print("----------------CON ADALINE--------------------")

# Re-etiquetar las clase de cada vector ejemplo en [-1,1]
yAd = np.where(Y == 0, -1, 1)
# Normalizar los vectores de características
XAd = (X - X.mean()) / X.std()
# Tasa de aprendizaje
eta=0.001
# Número de iteraciones
iter = 500
# Vector de pesos inicial
wAd = np.zeros(XAd.shape[1] + 1)
# Number of misclassifications
errors = []
# Cost function
costs = []

train_xAd = XAd[:50]
test_xAd = XAd[50:]
train_yAd = yAd[:50]
test_yAd = yAd[50:]

# Entrenamiento
for i in range(iter):
    output = np.dot(train_xAd, wAd[1:]) + wAd[0]
    errors = train_yAd - output
    wAd[1:] += eta * train_xAd.T.dot(errors)
    wAd[0] += eta * errors.sum()

# Prueba
errores = 0
for xi, target in zip(test_xAd, test_yAd) :
    activation = np.dot(xi, wAd[1:]) + wAd[0]
    output = np.where(activation >= 0.0, 1, -1)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_xAd), 
                                                        errores/len(test_xAd)*100))

#CON FUNCION DE ACTIVACION

X_train, X_test, y_train, y_test = train_test_split(
    datosX, datosY, test_size=.2)
clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', #identity  tanh
                    hidden_layer_sizes=(2,2,1), random_state=1, 
                    learning_rate_init=0.001, max_iter=5000)

clf.fit(X_train, y_train)                         

# Prueba
errores = 0
for xi, target in zip(X_test, y_test) :
    output = clf.predict(xi.reshape(1, -1))
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(X_test), 
                                                        errores/len(X_test)*100))












