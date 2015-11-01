import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

#read features from file for training

train = open('train.csv')
header = train.readline()
features=[]
labels=[]
for f in train.readlines():
	f=f.split(",")
	f=[int(x) for x in f]
	labels.append(f[0])
	features.append(f[1:])

features = np.array(features)
labels = np.array(labels)


features,test_features, labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(features, labels)
clf.score(test_features,test_labels)