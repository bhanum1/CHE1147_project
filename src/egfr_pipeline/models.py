"""Candidate estimators and reproducible search grids."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

def candidates():
    cs=[.001,.01,.1,1,10,100]
    weights=[None,'balanced']
    return {
      'baseline':(DummyClassifier(strategy='most_frequent'),{}),
      'logistic_l1':(LogisticRegression(solver='liblinear',l1_ratio=1,intercept_scaling=100,max_iter=10000,random_state=42),{'C':cs,'class_weight':weights}),
      'logistic_l2':(LogisticRegression(solver='lbfgs',l1_ratio=0,max_iter=10000),{'C':cs,'class_weight':weights}),
      'knn':(KNeighborsClassifier(n_jobs=1),{'n_neighbors':[3,5,11,21,41],'weights':['uniform','distance'],'p':[1,2]}),
      'random_forest':(RandomForestClassifier(n_estimators=300,n_jobs=1,random_state=42),{'max_depth':[None,15],'min_samples_leaf':[1,3],'max_features':['sqrt',.3],'class_weight':weights}),
      'svm_linear':(SVC(kernel='linear'),{'C':[.01,.1,1,10,100],'class_weight':weights}),
      'svm_rbf':(SVC(kernel='rbf'),{'C':[.1,1,10,100],'gamma':['scale',.001,.01],'class_weight':weights})}
