

from flask import Flask, request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
app = Flask(__name__)

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


from joblib import load, dump
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression




         
data = pd.read_csv('./datas/Maths.csv')


le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['address'] = le.fit_transform(data['address'])
data['guardian'] = le.fit_transform(data['guardian'])
data['schoolsup'] = le.fit_transform(data['schoolsup'])
data['famsup'] = le.fit_transform(data['famsup'])
data['school'] = le.fit_transform(data['school'])
data['famsup'] = le.fit_transform(data['famsup'])
data['paid'] = le.fit_transform(data['paid'])
data['romantic'] = le.fit_transform(data['romantic'])
data['famsize'] = le.fit_transform(data['famsize'])
data['reason'] = le.fit_transform(data['reason'])
data['Pstatus'] = le.fit_transform(data['Pstatus'])
data['higher'] = le.fit_transform(data['higher'])
data['internet'] = le.fit_transform(data['internet'])
data['nursery'] = le.fit_transform(data['nursery'])
data['Fjob'] = le.fit_transform(data['Fjob'])
data['Mjob'] = le.fit_transform(data['Mjob'])
data['activities'] = le.fit_transform(data['activities'])


X = data.drop('Dalc', axis=1)
Y = data['Dalc']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=200)


def DecisionTree():
    # data = request.get_json,
    # data = data['key']
    # data = data.split(",")
    # data = [data]
    
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    dump(model,'mymodelDecisionTree.pkl')

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    prediction = model.predict(data)
    print(prediction)
    print(accuracy)



def LinearRegressionFunc():
    X = np.array(data['Dalc']).reshape(-1,1)
    Y = np.array(data['Walc'])

    model = LinearRegression().fit(X,Y)
    dump(model,'mymodelLinearRegression.pkl')
    
    score = model.score(X,Y)
    print(score)



def KNeighborsClassifierFunc(): 
    knn = KNeighborsClassifier(n_neighbors=3)
    model = knn.fit(X_train, Y_train)
    
    dump(model,'mymodelKnn.pkl')
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(Y_test,y_pred)
    print(accuracy)
    
    