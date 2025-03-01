import numpy as np
from utils import *


class CrossValidation(object):
    """
    Cross Validation.

    This class is used to run cross validation on the data and plot the
    learning curve.

    """

    def __init__(self, k: int = 5) -> None:
        """
        Initialize the CrossValidation class by setting the value of k for
        k-fold CV.

        Args:
            k: default is 5

        Returns:
            None
        """

        self.k = k

    def create_folds(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     random_state: int = 42) -> list:
        """
        This function return a list [(X_1, y_1), (X_2, y_2), ..., (X_k, y_k)],
        X_i and y_i refer to the subset of X, y corresponding to fold i.
        Please note X_i, y_i here contain multiple data samples, but not a
        single one.

        Note: X and y will be numpy arrays of the same length.
        If self.k is < =  1 or > length of X, return [(X, y)].
        If the split is not perfect (i.e. len(X) % num_folds ! =  0), make the
        first folds the longer ones.

        Args:
            X: a numpy array of shape (N, D) containing the data
            y: a numpy array of shape (N,) containing the labels
            random_state: an integer used to seed the random generator

        Returns:
            folds: a list of k tuples (X_k, y_k) where k is the fold number

        Sample output:
            If there are 100 data samples with 5 features. When k = 5, 
            your output should be
            [(X_1, y_1), (X_2, y_2), (X_3, y_3), (X_4, y_4), (X_5, y_5)]
            Each X_i has shape (20, 5), y_i has shape (20, )
        """

        X, y = shuffle_data(X, y, random_state=random_state)

        if self.k < 1 or self.k > len(X):
            return [(X, y)]

        # >> YOUR CODE HERE
        fold_size = int(X.shape[0] / self.k)
        fold_size_first = fold_size
        if X.shape[0] % self.k != 0:
            fold_size_first = X.shape[0] % self.k + fold_size
        folds = []
        folds.append((X[0:fold_size_first], y[0:fold_size_first]))
        
        i = 1
        while i < self.k:
            folds.append((X[fold_size_first:fold_size_first+fold_size], y[fold_size_first:fold_size_first+fold_size]))
            fold_size_first += fold_size
            i += 1
        # << END OF YOUR CODE

        return folds

    def train_valid_split(self, folds: list, use_as_valid: int = 0) -> tuple:
        """
        This function sets the fold indexed by "use_as_valid" as the validation
        data and concatenate the remaining folds to use as training data.

        Args:
            folds: a list of folds [(X_1, y_1), (X_2, y_2), ..., (X_k, y_k)]
            use_as_valid: an integer indicating which fold to use as test

        Returns:
            (X_train, y_train), (X_test, y_test): the selected fold will be
                returned as the validation data and the remaining folds will be
                concatenated as training data
        """

        X_val, y_val = [], []
        X_train, y_train = [], []

        # >> YOUR CODE HERE
        X_val = folds[use_as_valid][0]
        y_val = folds[use_as_valid][1]

        i = 0
        while i < self.k:
            if i != use_as_valid:
                X_train.extend(folds[i][0])
                y_train.extend(folds[i][1])
            i += 1

        # << END OF YOUR CODE

        return (X_train, y_train), (X_val, y_val)

    def cross_val_score(self, clf, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        This function runs k-fold cross validation using the classifier "clf"
        on the dataset. 

        It will first split the data into k folds using the function
        "create_folds". Then, for each fold, it will use the remaining k-1
        folds as training data and the current fold as validation data. It will
        then train the classifier on the training data and evaluate it on the
        validation data. It will return the average accuracy calculated by
        calling the `score` function in the classifier.

        Args:
            clf: the classifier that you will be using. You can refer to 
                "evaluation.py"'s find_best_param function to understand what
                it can be.
            X: a numpy array of shape (N, D) containing the data
            y: a numpy array of shape (N,) containing the labels

        Returns: Accuracies for the training set and test set.
            train_accs: a list of shape (k, ), with each element representing 
                the accuracy on one split of the training set 
            test_accs: a list of shape (k, ), with each element representing 
                the accuracy on one split of the test set

        Example of returned values:
            train_accs = [0.5, 0.49, 0.51, 0.48, 0.47]
            val_accs = [0.5, 0.49, 0.51, 0.48, 0.47]
        """

        train_accs = []
        val_accs = []

        folds = self.create_folds(X, y)

        for i in range(self.k):
            # >>> YOUR CODE HERE (you may need to use the clf)
            (X_train, y_train), (X_val, y_val) = self.train_valid_split(folds, i)
            # print(i)
            # print(len(X_train), len(y_train), len(X_val), len(y_val)) #testing
            clf.fit(X, y)
            train_score = clf.score(X_train, y_train)
            val_score = clf.score(X_val, y_val)
            train_accs.append(train_score)
            val_accs.append(val_score)
            # << END OF YOUR CODE
            print(
                f'    fold {i} as val: train_acc={train_score:.5f}, val_acc={val_score:.5f}')

        return train_accs, val_accs