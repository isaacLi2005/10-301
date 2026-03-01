import numpy as np
import argparse
from argparse import ArgumentParser
import math 
import matplotlib.pyplot as plt 

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
    return np.array([0 for _ in range(1 + VECTOR_LEN)], dtype=float)

def fold_intercept_into_feature_vector(x: np.ndarray) -> np.ndarray: 
    """
    Takes a ndarray that is VECTOR_LEN long and returns one that is 
    1 + VECTOR_LEN long with a intercept 1 in prepended. 
    """
    return np.concatenate((np.array([1], dtype=float), x), axis=0)

def calculate_negative_log_likelihood(
    theta: np.ndarray, # shape (1 + VECTOR_LEN, )
    X: np.ndarray, # shape (N, VECTOR_LEN)
    y: np.ndarray, # shape (N,) 
) -> float: 
    
    assert(X.shape[0] == y.shape[0]) 

    sum = 0.0

    N = X.shape[0]
    for i in range(N): 
        p_i = sigmoid(np.dot(theta, fold_intercept_into_feature_vector(X[i])))

        if y[i] == 1: 
            sum += math.log(p_i) 
        else: 
            assert(y[i] == 0) 
            sum += math.log(1 - p_i)

    return -1 * (1 / N) * sum

def train(
    theta : np.ndarray, # shape (1 + VECTOR_LEN,)
    X : np.ndarray,     # shape (N, VECTOR_LEN)
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float, 
    validation_features: np.ndarray, # shape (N, VECTOR_LEN) 
    validation_labels: np.ndarray # shape (N,)
) -> np.ndarray:
    """
    Trains the entire thing. All datapoints. Returns negative log likeoihood on epochs. 
    """
    N = X.shape[0] # Number of datapoints. 

    train_nll = [] 
    validation_nll = []

    for _ in range(num_epoch): 
        for i in range(N): 
            x_i = fold_intercept_into_feature_vector(X[i]) 

            g = (sigmoid(np.dot(theta, x_i)) - y[i]) * x_i

            theta -= (learning_rate * g)

        train_nll.append(calculate_negative_log_likelihood(theta, X, y))
        validation_nll.append(calculate_negative_log_likelihood(theta, validation_features, validation_labels)) 

    return np.array(train_nll), np.array(validation_nll) 
        

def predict(
    theta : np.ndarray, # shape (1 + VECTOR_LEN,)
    X : np.ndarray      # shape (N, VECTOR_LEN)
) -> np.ndarray:
    """
    Predicts on all datapoints. 
    """
    result = [] 

    for x in X: 
        xi = fold_intercept_into_feature_vector(x) 

        predicted_probability = sigmoid(np.dot(theta, xi)) 

        if predicted_probability >= 0.5: 
            result.append(1)
        else:
            result.append(0)
    
    return np.array(result, dtype=int)



def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    """
    Generates the error across the entire dataset. 
    """
    assert(y_pred.shape == y.shape) 
    
    incorrect = 0 
    N = y.shape[0]
    for i in range(N): 
        if y_pred[i] != y[i]: 
            incorrect += 1
    
    return incorrect / N 

def read_input_tsv(input_file_path: str) -> tuple[np.ndarray, np.ndarray]: 
    """ 
    Reads a .tsv file into a numpy array so that our model may be run on it. 
    Returns a tuple of lavels and then features. 
    """
    data = np.genfromtxt(
        input_file_path, 
        delimiter = "\t", 
        dtype=float
    )

    y = data[:, 0].astype(int)
    X = data[:, 1:]

    return X, y

def write_output(output_path: str, predicted_labels: np.ndarray) -> None: 
    with open(output_path, "w", encoding='utf-8') as output_file: 
        for label in predicted_labels: 
            output_file.write(f"{label}\n")

def write_metrics(metric_path: str, train_error, test_error): 
    with open(metric_path, "w", encoding="utf-8") as metric_file: 
        metric_file.write(f"error(train): {train_error:.6f}\n")
        metric_file.write(f"error(test): {test_error:.6f}\n")

def graph_nll(train_nll: np.ndarray, validation_nll: np.ndarray, graph_path: str) -> None: 
    """
    Saves the negative log likelihoods to .pngs. 
    """
    assert(train_nll.shape[0] == validation_nll.shape[0])
    epochs = train_nll.shape[0]

    plt.figure() 
    plt.plot(np.array([i for i in range(1, epochs + 1)]), train_nll, label="training")
    plt.plot(np.array([i for i in range(1, epochs + 1)]), validation_nll, label="validation")
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Negative Log-Likelihood") 
    plt.legend() 
    plt.savefig(graph_path) 

def compare_learning_rates(
    train_features: np.ndarray, 
    train_labels: np.ndarray, 
    val_features: np.ndarray, 
    val_labels: np.ndarray, 
    num_epoch: int, 
    learning_rates: list[float], 
    out_path: str
): 
    nll_results = []
    for learning_rate in learning_rates: 
        theta = initialize_feature_vector() 

        train_nll, _ = train(
            theta, 
            train_features, 
            train_labels, 
            num_epoch, 
            learning_rate, 
            val_features, 
            val_labels
        )

        nll_results.append(train_nll) 

    plt.figure() 
    for i in range(len(nll_results)): 
        plt.plot(np.array([i for i in range(1, num_epoch + 1)]), nll_results[i], label=f"Learning rate of {learning_rates[i]}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Negative Log Likelihood") 
    plt.legend() 
    plt.savefig(out_path) 

def run(args: ArgumentParser) -> None: 
    """ 
    Uses the parsed command line inputs to load data, train on it, and then 
    """

    train_features, train_labels = read_input_tsv(args.train_input) 
    validation_features, validation_labels = read_input_tsv(args.validation_input) 
    test_features, test_labels = read_input_tsv(args.test_input) 

    theta = initialize_feature_vector() 
    train_nll, validation_nll = train(
        theta, 
        train_features, 
        train_labels, 
        args.num_epoch, 
        args.learning_rate, 
        validation_features, 
        validation_labels
    ) 
    graph_nll(train_nll, validation_nll, args.nll_graph_out)

    train_predictions = predict(theta, train_features) 
    test_predictions = predict(theta, test_features) 
    write_output(args.train_out, train_predictions) 
    write_output(args.test_out, test_predictions) 

    train_error = compute_error(train_predictions, train_labels) 
    test_error = compute_error(test_predictions, test_labels) 
    write_metrics(args.metrics_out, train_error, test_error)

    compare_learning_rates(
        train_features, train_labels, 
        validation_features, validation_labels, 
        1000, 
        [1e-1, 1e-2, 1e-3], 
        args.compare_lr_plot_out
    )





    


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
    parser.add_argument("nll_graph_out", type=str, help="Where to save the graph of training and validation nll")
    parser.add_argument("compare_lr_plot_out", type=str, help="Where to save the graph of different step sizes")
    args = parser.parse_args()

    run(args) 
