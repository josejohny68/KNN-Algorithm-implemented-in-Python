import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# importing the dataset
zoo=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 4 - KNN\\Zoo.csv")
zoo=zoo.drop("animal name",axis=1) # Animal name do not have that impact on classification
plt.hist(zoo.hair)# 0 and 1
plt.hist(zoo.feathers)# 0 and 1
plt.hist(zoo.eggs) # 0 and 1
plt.hist(zoo.milk)# 0 and 1
plt.hist(zoo.airborne)# 0 and 1
plt.hist(zoo.aquatic)# 0 and 1
plt.hist(zoo.predator) # 0 and 1 
plt.hist(zoo.toothed) # 0 and 1
plt.hist(zoo.backbone) # 0 and 1
plt.hist(zoo.breathes)# 0 and 1
plt.hist(zoo.venomous)# 0 and 1
plt.hist(zoo.fins) # 0 and 1
plt.hist(zoo.legs) # Numerical
plt.boxplot(zoo.tail)# 0 and 1
plt.hist(zoo.domestic)# 0 and 1
plt.hist(zoo.catsize)# 0 and 1

zoo.info()
 # There is no issue of scale
 # Checking if it is an imbalanced dataset
 plt.hist(zoo.type)# no issue of imbalanced
 
 # Checking if there are any null values
 zoo.isnull().sum()
 
 # Spliting the data into train dataset and test dataset
 from sklearn.model_selection import train_test_split
x=zoo.drop("type",axis=1)
y=zoo["type"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
# Building the KNN model
from sklearn import neighbors
model=neighbors.KNeighborsClassifier() 
model.fit(x_train,y_train)

# testing the model on test data
predict=model.predict(x_test)

# Checking the accuracy of the model
from sklearn.metrics import classification_report
classification_report(predict,y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(predict,y_test)

from sklearn.metrics import accuracy_score
accuracy_score(predict,y_test) # 90.47 %
