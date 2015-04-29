#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import cross_validation  
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit data with sklearn decision trees algorithm
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

# Get the accuracy
from sklearn.metrics import accuracy_score
prediction = clf.predict(features_test)
print "Prediction: ", prediction
print "Accuracy: ", accuracy_score(prediction, labels_test)
print "Number of POI's: ", np.count_nonzero(prediction)
print "People in Test Set: ", len(prediction) 
print "Accuracy if all zeros: ", accuracy_score([0]*29, labels_test)

from collections import Counter
confusion_matrix = Counter()

#truth = labels_test
prediction = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
truth = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
positives = [1]

binary_truth = [x in positives for x in truth]
binary_prediction = [x in positives for x in prediction]
for t, p in zip(binary_truth, binary_prediction):
    confusion_matrix[t,p] += 1

print confusion_matrix

from sklearn.metrics import precision_score
print "Precision Score: ", precision_score(prediction, truth)
from sklearn.metrics import recall_score
print "Recall Score: ", recall_score(prediction, truth)









