# KNN Algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Importing the dataset

glass=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 4 - KNN\\glass.csv")

glass.info()
plt.hist(glass.RI) # required variable for input
plt.hist(glass.Na)# required variable for input
plt.boxplot(glass.Mg)# required variable for input
plt.boxplot(glass.Al)# required variable for input
plt.boxplot(glass.Si)# required variable for input
plt.boxplot(glass.K)# required variable for input
plt.boxplot(glass.Ca)# required variable for input
plt.boxplot(glass.Ba)# required variable for input
plt.boxplot(glass.Fe)# required variable for input
# All input variables are in different ranges so before calculating the distance we need to bring them down to one scale
plt.hist(glass.Type)

# Spliting the data into train data and test data
x=glass.drop("Type",axis=1)
y=glass["Type"]

# Scale all input variables of x

X=preprocessing.scale(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

# Building the KNN Model

model=neighbors.KNeighborsClassifier()
model.fit(X_train,y_train)


# predicting on test data

prediction=model.predict(X_test)

# Checking the accuracy of the model

from sklearn.metrics import classification_report
classification_report(prediction,y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(prediction,y_test)

from sklearn.metrics import accuracy_score
accuracy_score(prediction,y_test)
# Model is 72% accuracy
