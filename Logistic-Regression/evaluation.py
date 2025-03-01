import os
from typing import Iterable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cross_validation import CrossValidation
from logistic_regression import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from utils import *


def find_best_param(X: np.ndarray,
                    y: np.ndarray,
                    k: int,
                    model_name: str,
                    param_name: str,
                    param_vals: Iterable) -> Tuple[dict, dict, Union[float, int]]:
    """
    Find the best parameter for a model using cross validation.

    Args:
        k: number of folds for cross validation
        model_name: name of the model
        param_name: name of the parameter
        param_vals: values of the hyperparameter

    Returns:
        train_accs_cv: a dictionary of training accuracies for each parameter
        val_accs_cv: a dictionary of validation accuracies for each parameter
        best_param: the best parameter value which results in the highest mean accuaracy
                    of the validation dataset

        An example for Logistc Regression (same format for other models):
            if lr = [0.1, 0.2, 0.3]
            then train_accs_cv could be {0.1: [0.8, 0.9, 0.7],
                                        0.2: [0.9, 0.9, 0.9],
                                        0.3: [0.7, 0.8, 0.9]}
            and val_accs_cv could be {0.1: [0.7, 0.8, 0.9],
                                    0.2: [0.9, 0.9, 0.9],
                                    0.3: [0.9, 0.8, 0.8]}
            then best_param should be 0.2
    """

    cv = CrossValidation(k=k)

    train_accs_cv = dict()
    val_accs_cv = dict()

    best_param = 0
    best_acc = 0

    for val in param_vals:
        if model_name == 'logistic':
            clf = LogisticRegression(lr=val)
        elif model_name == 'decision_tree':
            clf = DecisionTreeClassifier(max_depth=val)
        elif model_name == 'svm':
            clf = LinearSVC(C=val, dual=False, max_iter=10000)
        else:
            raise ValueError(f'Invalid model name {model_name}')

        print(f'{param_name} = {val}')

        # >>> YOUR CODE HERE
        CV = CrossValidation(k)
        train_accs, val_accs = CrossValidation.cross_val_score(CV, clf, X, y)
        train_accs_cv[val] = train_accs
        val_accs_cv[val] = val_accs

    for key in val_accs_cv:
        if best_acc < sum(val_accs_cv[key]):
            best_acc = sum(val_accs_cv[key])
            best_param = key
        # <<< END YOUR CODE

    return train_accs_cv, val_accs_cv, best_param


def plot_learning_curve(model_name: str,
                        params: dict,
                        train_accs_cv: dict,
                        val_accs_cv: dict,
                        filename: str) -> None:
    """
    Plot the learning curve for a model.

    Args:
        model_name: name of the model
        params: a dictionary of hyperparameters
        train_accs_cv: a dictionary of training accuracies for each parameter
        val_accs_cv: a dictionary of validation accuracies for each parameter
        filename: name of the file to save the plot

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    param_name, param_vals = list(params.keys())[0], list(params.values())[0]

    train_avgs = [np.mean(train_accs_cv[val]) for val in param_vals]
    train_stds = [np.std(train_accs_cv[val]) for val in param_vals]
    val_avgs = [np.mean(val_accs_cv[val]) for val in param_vals]
    val_stds = [np.std(val_accs_cv[val]) for val in param_vals]

    ax.errorbar(
        x=param_vals,
        y=train_avgs,
        yerr=train_stds,
        capsize=0.1,
        fmt="x-",
        label="Training",
    )
    ax.errorbar(
        x=param_vals,
        y=val_avgs,
        yerr=val_stds,
        capsize=0.1,
        fmt="o--",
        label="Validation",
    )
    ax.legend()
    ax.set_xticks([round(val, 5) for val in param_vals])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Accuracy")
    if model_name in ['logistic', 'svm']:
        ax.set_xscale('log')
    ax.set_title(f'Learning curve of {model_name}')

    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


def cv_logistic(X: np.ndarray, y: np.ndarray, k: int, lrs: Iterable) -> None:
    """
    Find the best learning rate for logistic regression using cross validation.

    Args:
        k: number of folds for cross validation
        lrs: learning rates

    Returns:
        None
    """
    train_accs_cv, val_accs_cv, best_lr = find_best_param(X, y,
                                                          k=k,
                                                          model_name='logistic',
                                                          param_name='lr',
                                                          param_vals=lrs)

    print(f'The best learning rate for Logistic Regression is {best_lr}')

    plot_learning_curve(model_name='logistic',
                        params={'lr': lrs},
                        train_accs_cv=train_accs_cv,
                        val_accs_cv=val_accs_cv,
                        filename='lr_curve.png')

def cv_dt(X: np.ndarray, y: np.ndarray, k: int, max_depths: Iterable) -> None:
    """
    Find the best max depth for decision tree using cross validation.

    Args:
        k: number of folds for cross validation
        max_depths: max depths

    Returns:
        None
    """

    (train_accs_cv, val_accs_cv, best_max_depth) = find_best_param(X, y,
                                                                   k=k,
                                                                   model_name='decision_tree',
                                                                   param_name='max_depth',
                                                                   param_vals=max_depths)

    print(f'The best max_depth for Decision Tree is {best_max_depth}')

    plot_learning_curve(model_name='decision_tree',
                        params={'max_depths': max_depths},
                        train_accs_cv=train_accs_cv,
                        val_accs_cv=val_accs_cv,
                        filename='dt_curve.png')


def cv_svm(X: np.ndarray, y: np.ndarray, k: int, Cs: Iterable) -> None:
    """
    Find the best C for SVM using cross validation.

    Args:
        k: number of folds for cross validation
        Cs: C values

    Returns:
        None
    """
    train_accs_cv, val_accs_cv, best_C = find_best_param(X, y,
                                                         k=k,
                                                         model_name='svm',
                                                         param_name='C',
                                                         param_vals=Cs)

    print(f'The best C for SVM is {best_C}')

    plot_learning_curve(model_name='svm',
                        params={'C': Cs},
                        train_accs_cv=train_accs_cv,
                        val_accs_cv=val_accs_cv,
                        filename='svm_curve.png')


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    # Load data
    X, y = load_data(os.path.join(
        os.path.dirname(__file__), 'dating_train.csv'))

    print('\n\n-------------CV: Logistic Regression-------------\n')
    lrs = np.logspace(-5, -1, 10)
    cv_logistic(X, y, k=5, lrs=lrs)

    print('\n\n-------------CV: Decision Tree-------------\n')
    max_depths = list(range(1, 11))
    cv_dt(X, y, k=5, max_depths=max_depths)

    print('\n\n-------------CV: SVM-------------\n')
    Cs = np.logspace(-2, 3, 10)
    cv_svm(X, y, k=5, Cs=Cs)

    print('\r\nDone.')