import numpy as np

class KNN():
    """
    k-nearest neighbors classifier。

    Shape D means the dimension of the feature.
    Shape N means the number of the training samples.

    Attributes:
        k: the number of nearest neighbors.
        X_train: the training samples.
        y_train: the labels of training samples.
    """

    def __init__(self, k: int = 3) -> None:
        """
        Constructor. Initialize by setting self.k to k.

        Args:
            k: the number of nearest neighbors.

        Returns:
            None.
        """

        # >> YOUR CODE HERE
        self.k = k
        # END OF YOUR CODE <<

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model by storing X_train and y_train to self.X_train and
        self.y_train.

        Args:
            X_train: the training samples of shape (N, D).
            y_train: the labels of training samples of shape (N,).

        Returns:
            None.
        """

        # >> YOUR CODE HERE
        self.X_train = X_train
        self.y_train = y_train
        # END OF YOUR CODE <<

    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the manhattan distance between x1 and x2.
        Args:
            x1: the first sample of shape (D,).
            x2: the second sample of shape (D,).

        Returns:
            manh_dist: the manhattan distance between x1 and x2.
        """

        # >> YOUR CODE HERE
        manh_dist = np.sum(np.abs(x1 - x2))
        # END OF YOUR CODE <<

        return manh_dist

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the test samples by calling
        predict_one_sample() for each sample. Store the predicted labels in
        y_pred of shape (N,).

        Args:
            X: the samples to be predicted of shape (N, D).

        Returns:
            Y_pred: the predicted labels of shape (N,).
        """

        # Initialization of Y_pred
        Y_pred = np.zeros(X.shape[0])

        # >> YOUR CODE HERE
        for i in range(X.shape[0]):
            print(f'\r\tPredicting {i+1}/{X.shape[0]}', end='')
            Y_pred[i] = self.predict_one_sample(X[i])
        print('')
        # END OF YOUR CODE <<
        return Y_pred

    def get_distances_to_other_samples(self, x: np.ndarray) -> np.ndarray:
        """
        Helper function to get the distances of the current sample to all
        other samples by calling manhattan_distance() for each sample.
        Store the distances in an array of shape (N,).

        Args:
            x: the sample to be predicted of shape (D,).

        Returns:
            dist: the distances of shape (N,).
        """
        dist = []

        # >> YOUR CODE HERE
        for i in range(self.X_train.shape[0]):
            dist.append(self.manhattan_distance(x, self.X_train[i]))
        dist = np.array(dist)        # END OF YOUR CODE <<

        return dist

    def vote(self, k_nearest: np.ndarray) -> int:
        """
        Vote for the label of the current sample by taking the most common
        label in the k nearest neighbors. Store the predicted label in
        y_pred of shape (1,). You may find np.bincount() and np.argmax()
        useful.

        Args:
            k_nearest: the k nearest neighbors of shape (k,).

        Returns:
            y_pred: the predicted label of shape (1,).
        """

        # >> YOUR CODE HERE
        y_pred = np.argmax(np.bincount(self.y_train[k_nearest]))
        # END OF YOUR CODE <<

        return y_pred

    def predict_one_sample(self, x: np.ndarray) -> int:
        """
        Predict the label of one sample. First get the distances of the
        current sample to all other samples by calling
        get_distances_to_other_samples(). Then get the k nearest neighbors
        by calling get_k_nearest_neighbors(). Finally, vote for the label of
        the current sample by calling vote(). Store the predicted label in
        y_pred of shape (1,).

        Args:
            x: the sample to be predicted of shape (D,).

        Returns:
            y_pred: the predicted label of shape (1,).
        """

        # >> YOUR CODE HERE
        distances = self.get_distances_to_other_samples(x)
        k_nearest = self.get_k_nearest_neighbors(distances)
        y_pred = self.vote(k_nearest)
        # END OF YOUR CODE <<

        return y_pred

    def get_k_nearest_neighbors(self, dist: np.ndarray) -> np.ndarray:
        """
        Get the k nearest neighbors of the current sample. You may find
        np.argsort() useful.

        Args:
            dist: the distances of shape (N,) of one sample.

        Returns:
            k_nearest: the k nearest neighbors of shape (k,).
        """

        # >> YOUR CODE HERE
        k_nearest = np.argsort(dist)[:self.k]
        # END OF YOUR CODE <<

        return k_nearest


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

import os
import pandas as pd
from data_split import *
import matplotlib.pyplot as plt

def load_phone_data(path: str, random_state=42):
    """
    Load the phone dataset.

    Args:
        path: path to the dataset

    Returns:
        X: data
        y: labels
    
    PLEASE DO NOT CHANGE!
    """

    df = pd.read_csv(path).sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    X = df.drop(columns=['price_range']).values
    y = df['price_range'].values

    print(
        f'Loaded data from {path}:\n\r X dimension: {X.shape}, y dimension: {y.shape}')

    return X, y

def accuracy(original, predictions):
    """
    Calculate the accuracy of given predictions on the given labels.

    Args:
        original: The original labels of shape (N,).
        predictions: Predictions of shape (N,).
        
    Returns:
        accuracy: The accuracy of the predictions.
    
    PLEASE DO NOT CHANGE!
    """
    return np.mean(original == predictions)
    
def evaluate_KNN() -> None:
    """
    Evaluate the KNN model on the data set.
    This function is supposed to do two things:
    1) Report the training and validation accuracies for 5-NN (k=5).
    2) Plot training and validation accuracies vs. value of k.
    PLEASE DO NOT CHANGE!
    """
    print('\n\n-------------KNN Performace (k=5) -------------\n')
    X, y = load_phone_data(os.path.join(
        os.path.dirname(__file__), r'dataset/phone_train.csv'))

    # Split the dataset into a training set and a validation set
    X_train, X_val, y_train, y_val = my_train_valid_split(X, y)

    # Initialize and fit a KNN model. 
    # Remember you could only use the training data to train the model, 
    # and validation data could not be seen！
    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    # Get predictions
    y_train_pred = knn.predict(X_train)
    y_val_pred = knn.predict(X_val)

    # Evaluate the KNN model on the training set.
    print(f'training acc: {accuracy(y_train, y_train_pred)}\nvalidation acc: {accuracy(y_val, y_val_pred)}')
    print('\n------------------------------------------\n')


    # Plot training and validation accuracies vs. value of k (1 to 10).
    K = list(range(1, 16))
    train_accs = []
    val_accs = []
    for k in K:
        print(f'k={k}:')
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        
        y_train_pred = knn.predict(X_train)
        y_val_pred = knn.predict(X_val)

        train_accs.append(accuracy(y_train, y_train_pred))
        val_accs.append(accuracy(y_val, y_val_pred))
    
    fig = plt.figure()
    plt.plot(K, train_accs, '-bo', label='training')
    plt.plot(K, val_accs, '-ro', label='validation')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.grid(True)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'knn_accs.png'))

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_KNN()