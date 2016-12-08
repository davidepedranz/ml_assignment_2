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


def generate_data(seed):
    """
    Generate a new classification problem with 2 classes and a lot of feature.
    :param seed: Seed to use for the random generator.
    :return: Pair (X, y) of features and classes.
    """
    m = 1000
    X, y = make_classification(n_samples=m, n_features=20, n_informative=10, n_redundant=3, n_repeated=2,
                               n_classes=2, random_state=seed)
    return X, y


def train_classifier(classifier, parameters, data, seed, k=10, plot=None):
    """
    Train the given classifier using a 10-fold cross validation.
    The given hyper-parameters are chosen using a 5-fold cross validation.
    :param classifier: Classifier to train.
    :param parameters: Hyper-parameters to choose from.
    :param data: Samples to use for the training and test.
    :param seed: Seed to use for the random generator, used for the k-fold split.
    :param k: Number of fold for the cross validation (default 10).
    :param plot: If not None, a plot for the hyper-parameters will be generated.
    :return: Performances (tuple of accuracy, f1, roc_auc) of the classifiers
             trained with the best hyper-parameters for each k-fold.
    """

    # save results of the 10-folds
    folds = []

    # store the performances of the best 10 trained classifiers
    accuracy = []
    f1 = []
    roc_auc = []

    # divide features and classes
    X, y = data

    # split the data for the 10-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        # extract training and test data
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # tune the hyper-parameters (5-fold cross validation, F1 scoring function)
        best_classifier = GridSearchCV(classifier, parameters, scoring='f1', cv=5, n_jobs=5, refit=True)
        best_classifier.fit(X_train, y_train)

        # TODO: measure performances of the different hyper parameters
        folds.append(best_classifier)

        # measure the performances
        prediction = best_classifier.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))
        f1.append(f1_score(y_test, prediction))
        roc_auc.append(roc_auc_score(y_test, prediction))

    # plot the choice of the best hyper-parameters
    if plot is not None:

        # average for the 10-folds
        # np.mean(np.array(list(map(lambda x: x.cv_results_['mean_test_score'], folds))), axis=0)

        # extract the parameter
        if len(parameters.keys()) != 1:
            raise NotImplementedError("The number of hyper-parameters is not equal to 1. "
                                      "I do not know how to plot them.")
        parameter_name = list(parameters.keys())[0]

        # extract the hyper-parameter values
        assert len(folds) == k
        x_axis_values = folds[0].cv_results_['param_' + parameter_name]
        for i in range(k):
            np.testing.assert_array_equal(x_axis_values, folds[i].cv_results_['param_' + parameter_name])

        # extract the F1 scores
        y_axis_values = list(map(lambda x: x.cv_results_['mean_test_score'], folds))

        # plot the graph
        for i in range(k):
            plt.semilogx(x_axis_values, y_axis_values[i], label=i + 1)
        plt.title(plot + " - tuning of the best value for " + parameter_name)
        plt.xlabel(parameter_name + " parameter")
        plt.ylabel("F1 scores (on the test set)")
        plt.legend(loc=4)
        plt.savefig(plot.lower().replace(" ", "_") + '.png')
        plt.close()

    # return the performances
    return accuracy, f1, roc_auc


def print_performances(classifier, performances):
    """
    Print the performances of a 10-fold cross validation in a structured way.
    :param classifier: Classifier type used to obtain the performances.
    :param performances: Tuple of ist of accuracy scores, F1 scores and ROC AUC scores.
    """

    # extract the performances of this classifier
    accuracy, f1, roc_auc = performances

    # make sure the input is correct
    l1, l2, l3 = len(accuracy), len(f1), len(roc_auc)
    assert l1 == l2 == l3

    # print each row in this format, so that to make a table
    columns = "   {:0.3f}  |  {:0.3f} |  {:0.3f} "
    row_format = "  {:2d}  |" + columns
    mean_format = " mean |" + columns
    separator = '------+----------+--------+---------'

    # the classifier
    print('\n-----------------------------------------')
    print(' Performances for: ' + classifier)
    print('-----------------------------------------\n')

    # header
    print(separator)
    print('   i  | accuracy |   f1   | roc auc ')
    print(separator)

    # body
    for i in range(l1):
        print(row_format.format(i + 1, accuracy[i], f1[i], roc_auc[i]))
    print(separator)

    # mean
    print(mean_format.format(np.mean(accuracy), np.mean(f1), np.mean(roc_auc)))
    print(separator + '\n')


def main():
    """
    Compare the performances of SVM, Random Forest and Naive Bayes
    on a randomly generated classification problem.
    """

    # TODO: remove
    k = 4

    # TODO: read it from the parameters of the script
    # pick some random seed
    seed = 17294

    # generate the samples to work with
    X, y = generate_data(seed)

    # train SVM
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm_parameters = {'C': [0.01, 0.033, 0.1, 0.33, 1, 3.33, 10, 33, 100]}
    svm_performances = train_classifier(svm, svm_parameters, data=(X, y), seed=seed, k=k, plot='SVM')
    print_performances('SVM', svm_performances)

    # train Random Forest
    rf = RandomForestClassifier(random_state=seed)
    rf_parameters = {'n_estimators': [10, 100, 1000]}
    rf_performances = train_classifier(rf, rf_parameters, data=(X, y), seed=seed, k=k, plot='Random Forest')
    print_performances('Random Forest', rf_performances)

    # train Naive Bayes
    nb = GaussianNB()
    nb_parameters = {}
    nb_performances = train_classifier(nb, nb_parameters, data=(X, y), seed=seed, k=k)
    print_performances('Naive Bayes', nb_performances)

    # evaluate performances


# Entry Point
if __name__ == "__main__":
    main()
