import numpy as np
from utils import *


class LogisticRegression(object):
    """
    Logistic regression.

    Shape D means the dimension of the feature.
    Shape N means the number of the training examples.

    Attributes:
        weights: The weight vector of shape (D, 1).
        bias: The bias term.
    """

    def __init__(self,
                 lr: float = 1e-3,
                 max_epochs: float = 5000,
                 tol: float = 1e-2) -> None:
        """
        Initialize the parameters of the logistic regression by setting parameters
        weights and bias to None

        Args:
            lr: The learning rate, default is 0.001.
            max_epochs: The maximum number of epochs, default is 5000.
            tol: The tolerance for the loss, default is 1e-2.

        Returns:
            None
        """

        self.weights = None
        self.bias = None
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z: The input of shape (N,).

        Returns:
            sigmoid: The sigmoid output of shape (N,).
        """

        # Set threshold to avoid overflow
        threshold = 500
        z = np.clip(z, -threshold, threshold)

        sigmoid = 1.0 / (1 + np.exp(-z))

        return sigmoid

    def gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """
        Compute the gradient of the loss with respect to the weights by first
        computing the probabilities of the labels for training data and then
        computing the gradient with respect to the weights.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).

        Returns:
            grad_w: The gradient of the loss with respect to the weights of
                shape (D, 1).
            grad_b: The gradient of the loss with respect to the bias of
                shape (1,).
        """

        y_pred_proba = self.predict_proba(X_train)
        grad_w = np.dot(X_train.T, (y_pred_proba - y_train)) / y_train.shape[0]
        grad_b = np.mean(y_pred_proba - y_train)

        return grad_w, grad_b

    def logistic_loss(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Compute the loss of the logistic regression by first computing the
        probabilities of the labels for training data and then computing the
        loss.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).

        Returns:
            logistic_loss: The logistic loss.
        """

        y_pred_proba = self.predict_proba(X_train)
        logistic_loss = np.mean(-y_train * np.log(y_pred_proba) -
                                (1 - y_train) * np.log(1 - y_pred_proba))

        return logistic_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability of the labels given the features, using the
        weights and bias of the logistic regression and the sigmoid function.

        Args:
            X: The features of shape (N, D).

        Returns:
            y_pred_proba: The probabilities of the labels of shape (N,).
        """

        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self.sigmoid(z)

        return y_pred_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions given new inputs by first computing the probabilities
        and then rounding them to the closest integer. 

        Args:
            X: The features of shape (N, D).

        Returns:
            y_pred: Predicted labels of shape (N,).
        """

        y_pred_proba = self.predict_proba(X)
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)

        return y_pred

    def train_one_epoch(self) -> None:
        """
        Train the logistic regression for one epoch. First compute the
        the gradients with respect to the weights and bias, and then update the
        weights and bias.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).
            learning_rate: The learning rate, default is 0.001.

        Returns:
            grad_w: The gradient of the loss with respect to the weights of
                shape (D, 1).
            grad_b: The gradient of the loss with respect to the bias of
                shape (1,).
        """

        grad_w, grad_b = self.gradient(self.X, self.y)
        new_weights = self.weights - self.lr * grad_w
        new_bias = self.bias - self.lr * grad_b
        self.weights, self.bias = new_weights, new_bias

        return grad_w, grad_b

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy.
        Args:
            y_true: The true labels of shape (N,).
            y_pred: Predictions for the features of shape (N,).

        Raises:

            ValueError: If the shape of y_true or y_pred is not matched.
        Returns:
            The accuracy of the logistic regression.
        """

        if len(y_true) != len(y_pred):
            raise ValueError('y_true and y_pred have mismatched lengths.')

        return np.mean(y_pred == y_true)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression by training it for a number of epochs.

        First save `X` and `y` as attributes of the class, and then initialize
        the weights and bias, and then train the logistic regression by calling
        the `train_one_epoch` function for `max_epochs` times. If the gradient
        of weights is smaller than `tol`, stop training.


        Args:
            X: training data
            y: training labels

        Returns:
            None.
        """
        self.X = X
        self.y = y

        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()

        for _ in range(self.max_epochs):
            grad_w, _ = self.train_one_epoch()
            if (np.abs(grad_w) < self.tol).all():
                break

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy.
        Args:
            X: The features of shape (N, D).
            y: The labels of shape (N,).

        Returns:
            The accuracy of the logistic regression.
        """
        return self.accuracy(y, self.predict(X))
