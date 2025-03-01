import os
from random import seed
import numpy as np
from random import randint


class NumpyBasics:
    """Class include some numpy basic operations."""

    def create_zero_matrix(rows: int, cols: int):
        """Create a zero matrix of the given size.

        You may assume both rows and cols are non-negative.

        Args:
            rows:
                Number of rows in the matrix.
            cols:
                Number of columns in the matrix.

        Returns:
            A 2-D numpy matrix of the given size. For example:

            [[0. 0. 0.]
            [0. 0. 0.]]
        """

        # >> YOUR CODE HERE
        matrix = np.zeros((rows, cols))
        return matrix
        # END OF YOUR CODE

    def create_vector(n: int):
        """Create a vector of the given size.

        Fill it with random integer values between 0 and square size. You may
        assume size is non-negative.

        Args:
            n:
                Number of elements in the vector.

        Returns:
            A 1-D numpy array of the given size. For example:

            [20 3 0 23 8]
        """

        # >> YOUR CODE HERE
        return np.array([randint(0, n**2) for _ in range(n)])
        # END OF YOUR CODE

    def calculate_matrix_inverse(matrix: np.ndarray):
        """Calculate the inverse of the given matrix.

        You may assume the matrix is square and has a determinant != 0.

        Args:
            matrix:
                A 2-D numpy matrix.

        Returns:
            A 2-D numpy array of the inverse of the given matrix. For example:

            [[-2.   1. ]
            [ 1.5 -0.5]]
        """

        # >> YOUR CODE HERE
        return np.linalg.inv(matrix)
        # END OF YOUR CODE

    def calculate_dot_product(vector1: np.array, vector2: np.array):
        """Calculate the dot product of the given vectors.

        You may assume the vectors are 1-D arrays.

        Args:
            vector1:
                A 1-D numpy array.
            vector2:
                A 1-D numpy array.

        Returns:
            The dot product of the given vectors. 
        """

        # >> YOUR CODE HERE
        return np.dot(vector1, vector2)
        # END OF YOUR CODE

    def solve_linear_system(A: np.ndarray, b: np.ndarray):
        """Solve the linear system described by Ax = b.

        You may assume A is square and has a determinant != 0. You may also
        assume there is one and only one solution.

        Args:
            A:
                A 2-D numpy matrix.
            b:
                A 1-D numpy array.

        Returns:
            A 1-D numpy array of the solution to the linear system.
        """

        # >> YOUR CODE HERE
        return np.dot(np.linalg.inv(A), b)
        # END OF YOUR CODE


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""


def test(func, expected_output, **kwargs) -> bool:
    """
    Test a function with some inputs.

    Args:
        func: The function to test.
        expected_output: The expected output of the function.
        **kwargs: The arguments to pass to the function.

    Returns:
        True if the function outputs the expected output, False otherwise.
    """
    output = func(**kwargs)

    try:
        assert np.allclose(output, expected_output)
        print(f'Testing {func.__name__}: passed')
        return True
    except AssertionError:
        print(f'Testing {func.__name__}: failed')
        print(f'Expected:\n {expected_output}')
        print(f'Got:\n {output}')
        return False


def evaluate_numpy_basics():
    """
    Test your implementation in numpy_basics.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Numpy Basics-------------\n')
    print('This test is not exhaustive by any means. You should test your ')
    print('implementation by yourself.\n')
    print('If you have failed create_vector() test, please check if you ')
    print('have used randint() from random, not np.random.randint(), ')
    print('to generate random numbers.\n')

    test(NumpyBasics.create_zero_matrix, np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), rows=3, cols=4)

    seed(42)
    test(NumpyBasics.create_vector, np.array([20, 3, 0, 23, 8]), n=5)

    test(NumpyBasics.calculate_matrix_inverse, np.array(
        [[-2, 1], [1.5, -0.5]]), matrix=np.array([[1, 2], [3, 4]]))

    test(NumpyBasics.calculate_dot_product, 32,
         vector1=np.array([1, 2, 3]), vector2=np.array([4, 5, 6]))

    test(NumpyBasics.solve_linear_system, np.array(
        [-4, 4.5]), A=np.array([[1, 2], [3, 4]]), b=np.array([5, 6]))


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_numpy_basics()
