import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

# fix a seed to make this experiment repeatable
seed = 0

# create the dataset
m = 1000
X, y = make_classification(n_samples=m, n_features=20, n_informative=10, n_redundant=3, n_repeated=2,
                           n_classes=2, random_state=seed)

# container for the performances
performances = {
    'svm': {'accuracy': [], 'f1': [], 'roc_auc': []},
    'rf': {'accuracy': [], 'f1': [], 'roc_auc': []},
    'bn': {'accuracy': [], 'f1': [], 'roc_auc': []}
}

# make 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X, y):
    # extract training and test data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # SVM
    svm = GridSearchCV(estimator=SVC(kernel='rbf', class_weight='balanced'),
                       param_grid={'C': np.arange(0.1, 1, 0.1)}, scoring='f1', cv=5, n_jobs=5, refit=True)
    svm.fit(X_train, y_train)
    svm_prediction = svm.predict(X_test)

    performances['svm']['accuracy'].append(accuracy_score(y_test, svm_prediction))
    performances['svm']['f1'].append(f1_score(y_test, svm_prediction))
    performances['svm']['roc_auc'].append(roc_auc_score(y_test, svm_prediction))

    # TODO: same for the other 2

print(performances)

# TODO: take averages, select best (maybe small graph?)
