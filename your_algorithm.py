#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
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
################################################################################
# -*- coding: utf-8 -*-

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

#sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform',
#algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

from sklearn import neighbors
clf=neighbors.KNeighborsClassifier(n_neighbors=10,weights='distance')
clf.fit(features_train, labels_train)
acc_knn=clf.score(features_test, labels_test)
print acc_knn

#sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50,
#learning_rate=1.0, algorithm='SAMME.R', random_state=None)

from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(n_estimators=50,algorithm='SAMME')
clf.fit(features_train, labels_train)
acc_ada=clf.score(features_test, labels_test)
print acc_ada

#class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,
#min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
#max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
#oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=15,max_features=1,max_depth=None,min_samples_split=2,bootstrap=True)
clf.fit(features_train, labels_train)
acc_for=clf.score(features_test, labels_test)


print acc_for

#prettyPicture(clf, features_test, labels_test)
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
#output_image("test.png","png",open("test.png","rb").read())






