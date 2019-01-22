# some observations, each row is known as an observation
# each column is a feature

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importinng the iris dataset from scikit learn datasets itself
from sklearn.datasets import load_iris
 
# making the object of our dataset
iris = load_iris()

iris.feature_names
iris.target_names

# deviding the dataset
X = iris.data
y = iris.target

unique = np.unique(y)

# splitting up the datasets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)

# fitting  our model to KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
train_acc = []
test_acc = []
for i in range(1, 11):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(Xtrain, ytrain)
    train_acc.append(classifier.score(Xtrain, ytrain))
    test_acc.append(classifier.score(Xtest, ytest))
    
plt.plot(range(1, 11), train_acc, label = 'Training accuracy')
plt.plot(range(1, 11), test_acc,label = 'Test accuracy')
plt.xlabel('no. of neghbors')
plt.ylabel('Accuracies')
plt.legend()
plt.show()

# taking k as 5 which is by default also
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X, y)

# predicting the accuracy
y_pred = classifier.predict(Xtest)

from sklearn.metrics import r2_score
r2_score(ytest, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, y_pred)

