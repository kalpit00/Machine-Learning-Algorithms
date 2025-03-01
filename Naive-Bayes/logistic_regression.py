import os

import numpy as np
from matplotlib import pyplot as plt

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

    def __init__(self) -> None:
        """
        Initialize the parameters of the logistic regression by setting parameters
        weights and bias to None

        Args:
            None.

        Returns:
            None.
        """
        self.weights = None
        self.bias = None
    

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z: The input of shape (N,).

        Returns:
            sigmoid: The sigmoid output of shape (N,).
        """

        # >> YOUR CODE HERE
        sigmoid = 1.0 / (1 + np.exp(-z))
        # << END OF YOUR CODE

        # The following part is for avoiding the value to be 0 or 1. DO NOT MODIFY.
        sigmoid[sigmoid > 0.99] = 0.99
        sigmoid[sigmoid < 0.01] = 0.01
        return sigmoid

    def gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
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

        grad_w, grad_b = None, None

        # >> YOUR CODE HERE
        y_pred_proba = self.predict_proba(X_train)
        grad_w = np.dot(X_train.T, (y_pred_proba - y_train)) / y_train.shape[0]
        grad_b = np.mean(y_pred_proba - y_train)        
        # << END OF YOUR CODE

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
        
        # >> YOUR CODE HERE
        y_pred_proba = self.predict_proba(X_train)
        logistic_loss = np.mean(-y_train * np.log(y_pred_proba) -
                                (1 - y_train) * np.log(1 - y_pred_proba))        # << END OF YOUR CODE

        return logistic_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability of the labels given the features, using the
        weights and bias of the logistic regression and the sigmoid function.

        Args:
            X: The features of shape (N, D).

        Returns:
            y_pred_proba: The probabilities of the labels in numpy array with shape (N,).
        """
        
        # >> YOUR CODE HERE
        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self.sigmoid(z)        
        # << END OF YOUR CODE

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

        # >> YOUR CODE HERE
        y_pred_proba = self.predict_proba(X)
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)        
        # << END OF YOUR CODE

        return y_pred

    def train_one_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                        learning_rate: float = 0.001) -> None:
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

        grad_w, grad_b = None, None

        # >> YOUR CODE HERE
        grad_w, grad_b = self.gradient(X_train, y_train)
        new_weights = self.weights - learning_rate * grad_w
        new_bias = self.bias - learning_rate * grad_b
        self.weights, self.bias = new_weights, new_bias        
        # << END OF YOUR CODE

        return grad_w, grad_b

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_valid: np.ndarray,
              y_valid: np.ndarray,
              max_epochs: int = 50000,
              lr: float = 0.001,
              tol: float = 1e-2) -> None:
        """
        Train the logistic regression using gradient descent. First initialize
        the weights and bias, and then train the model for max_epochs iterations
        by calling train_one_epoch() If the absolute value of the gradient of
        weights is less than tol, stop training. You may use self.logistic_loss()
        and accuracy() (in utils.py) to compute the loss and accuracy of the model and 
        print them out during training.


        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).
            max_epochs: The maximum number of epochs, default is 50000.
            lr: The learning rate, default is 0.001.
            tol: The tolerance for early stopping, default is 1e-2.

        Returns:
            None.
        """

        self.weights = np.random.randn(X_train.shape[1])
        self.bias = np.random.randn()

        for epoch in range(max_epochs):

            # >> YOUR CODE HERE
            grad_w, _ = self.train_one_epoch(X_train, y_train, lr)

            train_loss = self.logistic_loss(X_train, y_train)
            valid_loss = self.logistic_loss(X_valid, y_valid)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = accuracy(y_train, y_train_pred)
            valid_acc = accuracy(y_valid, y_valid_pred)
            # << END OF YOUR CODE

            if epoch % 100 == 0:
                print(
                    f'Epoch {epoch}: train loss = {train_loss:.8f}, valid loss = {valid_loss:.8f}, train acc = {train_acc:.8f}, valid acc = {valid_acc:.8f}')

        print(
            f'Final: train loss = {train_loss:.8f}, valid loss = {valid_loss:.8f}, train acc = {train_acc:.8f}, valid acc = {valid_acc:.8f}')


def plot_logistic_regression_curve(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_valid: np.ndarray,
                                   y_valid: np.ndarray,
                                   max_epochs: int = 50000,
                                   lr: float = 0.001,
                                   ) -> None:
    """
    Plot loss and accuracy curves for the logistic regression classifier.

    Args:
        logistic_regression_classifier: The logistic regression classifier.
        X_train: The training features of shape (N, D).
        y_train: The training labels of shape (N,).
        X_valid: The validation features of shape (N, D).
        y_valid: The validation labels of shape (N,).
        max_epochs: The maximum number of epochs.
        lr: The learning rate.
        tol: The tolerance for early stopping.

    Returns:
        None
    """
    lr_classifier = LogisticRegression()
    lr_classifier.weights = np.random.randn(X_train.shape[1])
    lr_classifier.bias = np.random.randn()

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for _ in range(max_epochs):

        ## >>> YOUR CODE HERE
        lr_classifier.train_one_epoch(X_train, y_train, learning_rate=lr)
        train_loss.append(lr_classifier.logistic_loss(X_train, y_train))
        valid_loss.append(lr_classifier.logistic_loss(X_valid, y_valid))
        train_acc.append(accuracy(y_train, lr_classifier.predict(X_train)))
        valid_acc.append(accuracy(y_valid, lr_classifier.predict(X_valid)))        
        ## <<< END OF YOUR CODE

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(os.path.dirname(
        __file__), "learning_curve_lr.png"))

"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

def evaluate_lr():
    X, y = load_data(os.path.join(
        os.path.dirname(__file__), 'dataset/dating_train.csv'))
    X_train, X_valid, y_train, y_valid = my_train_test_split(
        X, y, 0.2, random_state=42)

    print('\n\n-------------Fitting Logistic Regression-------------\n')
    lr = LogisticRegression()
    lr.train(X_train, y_train, X_valid,
             y_valid, max_epochs=12000, lr=0.001)

    print('\n\n-------------Logistic Regression Performace-------------\n')
    evaluate(y_train,
             lr.predict(X_train),
             y_valid,
             lr.predict(X_valid))

    print('\n\n-------------Plotting learning curves-------------\n')

    print('Plotting Logistic Regression learning curves...')

    plot_logistic_regression_curve(X_train, y_train, X_valid, y_valid, max_epochs=12000, lr=0.001)

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    print('\n-------------Logistic Regression-------------')
    evaluate_lr()

    print('\nDone!')
