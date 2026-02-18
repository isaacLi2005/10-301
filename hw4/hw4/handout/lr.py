import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt



def sigmoid(x : np.ndarray) -> np.ndarray:
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def initialize_feature_vector() -> np.ndarray:
    """
    Initializes a vector representing omega with all zeroes. 
    The vector is of dimension (301, ), one intercept and the 300 glove terms. 
    """
    return np.array([0 for _ in range(1 + VECTOR_LEN)])

def fold_intercept_into_feature_vector(x: np.ndarray) -> np.ndarray: 
    """
    Takes a ndarray that is VECTOR_LEN long and returns one that is 
    1 + VECTOR_LEN long with a intercept 1 in prepended. 
    """
    return np.concat((np.array([1]), x), axis=0)


def train(
    theta : np.ndarray, # shape (?,)
    X : np.ndarray,     # shape (?, ?)
    y : np.ndarray,     # shape (?,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    pass


def predict(
    theta : np.ndarray, # shape (?,)
    X : np.ndarray      # shape (?, ?)
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    pass


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    pass


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()
