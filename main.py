import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# TODO: take averages, select best (maybe small graph?)
def show_results(performances):
    for key, performance in performances.items():
        print('')
        print(' --> ' + key)
        print('')
        print('----+----------+--------+---------')
        print('  i | accuracy |   f1   | roc auc ')
        print('----+----------+--------+---------')

        row_format = " {:2d} |   {:0.3f}  |  {:0.3f} |  {:0.3f} "
        for i in range(10):
            print(row_format.format(i + 1, performance['accuracy'][i], performance['f1'][i], performance['roc_auc'][i]))
        print('----+----------+--------+---------')
        print(row_format.format(0, np.mean(performance['accuracy']), np.mean(performance['f1']),
                                np.mean(performance['roc_auc'])))
        print('----+----------+--------+---------')


def tune_hyper_parameter(estimator, param_grid, X_train, y_train):
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='f1', cv=5, n_jobs=5, refit=True)
    clf.fit(X_train, y_train)
    return clf


def measure_performance(classifier, X_test, y_test, name, performances):
    # make the prediction
    prediction = classifier.predict(X_test)

    # measure the performances
    performances[name]['accuracy'].append(accuracy_score(y_test, prediction))
    performances[name]['f1'].append(f1_score(y_test, prediction))
    performances[name]['roc_auc'].append(roc_auc_score(y_test, prediction))


def train_classifier(kf_split, ):
    pass


# noinspection PyPep8Naming
def main():
    # fix a seed to make this experiment repeatable
    seed = 123823

    # create the dataset
    m = 1000
    X, y = make_classification(n_samples=m, n_features=20, n_informative=10, n_redundant=3, n_repeated=2,
                               n_classes=2, random_state=seed)

    # container for the performances
    performances = {
        'svm': {'accuracy': [], 'f1': [], 'roc_auc': []},
        'rf': {'accuracy': [], 'f1': [], 'roc_auc': []},
        'nb': {'accuracy': [], 'f1': [], 'roc_auc': []},
        'lr': {'accuracy': [], 'f1': [], 'roc_auc': []}
    }

    # make 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        # extract training and test data
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # SVM
        svm = tune_hyper_parameter(SVC(kernel='rbf', class_weight='balanced'),
                                   {'C': [0.01, 0.033, 0.1, 0.33, 1, 3.33, 10, 33, 100]}, X_train, y_train)
        measure_performance(svm, X_test, y_test, 'svm', performances)

        # set the axes
        # ax.set_xlim((min_x, max_x))
        # ax.set_ylim((min_y, max_y))

        plt.semilogx(svm.cv_results_['param_C'], svm.cv_results_['mean_test_score'])
        plt.xlabel("C parameter")
        plt.ylabel("f1")
        plt.savefig('output/svm_' + str(i + 1) + '.png')

        # random forest
        rf = tune_hyper_parameter(RandomForestClassifier(random_state=seed),
                                  {'n_estimators': [10, 100, 1000]}, X_train, y_train)
        measure_performance(rf, X_test, y_test, 'rf', performances)

        # naive bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        measure_performance(nb, X_test, y_test, 'nb', performances)

        # logistic regression
        lr = tune_hyper_parameter(LogisticRegression(), {'C': [1e-02, 1e-01, 1e00, 1e01, 1e02]}, X_train, y_train)
        measure_performance(lr, X_test, y_test, 'lr', performances)

    # final check
    show_results(performances)


if __name__ == "__main__":
    main()
