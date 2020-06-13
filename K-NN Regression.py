# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:14:19 2019

@author: ertasnihan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#eksik verilerin temizlenmesi(sayısal)
veriler = pd.read_csv('veriler.csv')
print(veriler)

x= veriler.iloc[:,1:4].values  # bağımsız değişkenler
y= veriler.iloc[:,4:].values  # bağımlı değişkenler

#verilerin eğitim ve test kümesi olarak bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

#öznitelik ölçekleme(standartlaşma)
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(x_train) #öğren ve dönüştür
X_test = sc.transform(x_test)  #dönüştür


#Logistic regression classification
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred =logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)  # test ettiğimiz veri ve tahmin edilen değerleri gösterir
print(cm)

 # K- NN classification
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1, metric='minkowski') # 1 komşu
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)









    
    









