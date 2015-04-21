#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
'''
# K-Nearest-Neighbors Classifer
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=22)
t0 = time()
clf.fit(features_train,labels_train)
print "Training: ", round(time() - t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "Prediction: ", round(time() - t1, 3), "s"
print "Accuracy: ", accuracy_score(pred, labels_test)
# Accuracy: 0.944
# Training time: 0.001s
# Prediction time: 0.002s
'''
'''
# AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50)
t0 = time()
clf.fit(features_train, labels_train)
print "Training: ", round(time() - t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "Prediction: ", round(time() - t1, 3), "s"
print accuracy_score(pred, labels_test)
# Accuracy: 0.924
# Training time: 0.072s
# Prediction time: 0.007s
'''
# Random Forests Classifer
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=19)
t0 = time()
clf = clf.fit(features_train, labels_train)
print "Training: ", round(time() - t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "Prediction: ", round(time() - t1, 3), "s"
print "Accuracy: ", accuracy_score(pred, labels_test)
# Accuracy: 0.928
# Training time: 0.022s
# Prediction time: 0.003s

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass