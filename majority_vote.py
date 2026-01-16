import sys 
import numpy as np 
from typing import Iterable, Sequence

class CommandInput:
    """
    A data class that will store what the user inputted to run this script. 
    """
    def __init__(self, 
                 train_input: str, 
                 test_input: str, 
                 train_out: str, 
                 test_out: str, 
                 metrics_out: str):
        self.train_input = train_input 
        self.test_input = test_input 
        self.train_out = train_out 
        self.test_out = test_out 
        self.metrics_out = metrics_out 

def get_last_column(array): 
    """
    Finds and returns the last column of a numpy array, assuming its named. 
    """
    return array[array.dtype.names[-1]]

class MajorityVoteClassifier:
    """
    A classifier that will predict whatever label appeared the most within 
    the training data. 
    """
    def __init__(self):
        self.majority_vote = None 

    def train(self, command_input: CommandInput): 
        """
        Note that we assume the class label is at the last column. 
        """
        training_data = np.genfromtxt(
            command_input.train_input, 
            delimiter = "\t", 
            names = True, 
            dtype = int
        )
        label_column = get_last_column(training_data)

        num_0 = 0 
        num_1 = 0 
        for label in label_column: 
            if label == 0:
                num_0 += 1
            else:
                assert(label == 1)
                num_1 += 1 
        
        if num_1 >= num_0: 
            self.majority_vote = 1
        else:
            self.majority_vote = 0 
    
    def predict(self, file_name: str) -> list[int]:
        assert(self.majority_vote is not None)

        input_data = np.genfromtxt(
            file_name, 
            delimiter = "\t", 
            names = True, 
            dtype = int
        ) 
        num_rows = input_data.shape[0]

        return [self.majority_vote for _ in range(num_rows)]
    
def write_outputs_and_metrics(command_input: CommandInput, 
                              classifier: MajorityVoteClassifier): 
    """
    Writes the output .txt files for the training data and testing data. 
    """
    training_predictions = classifier.predict(command_input.train_input)
    testing_predictions = classifier.predict(command_input.test_input) 

    # Writing the raw outputs. 
    with open(command_input.train_out, "w") as train_out_file: 
        for training_prediction in training_predictions: 
            train_out_file.write(f"{training_prediction}\n")
    with open(command_input.test_out, "w") as test_out_file: 
        for testing_prediction in testing_predictions: 
            test_out_file.write(f"{testing_prediction}\n")



    # Calculating the error rate. 
    training_data = np.genfromtxt(
        command_input.train_input, 
        delimiter = "\t", 
        names = True, 
        dtype = int
    )
    testing_data = np.genfromtxt(
        command_input.test_input, 
        delimiter = "\t", 
        names = True, 
        dtype = int
    )

    train_labels_column = get_last_column(training_data)
    test_labels_column = get_last_column(testing_data)

    assert(len(train_labels_column) == len(training_predictions))
    assert(len(test_labels_column) == len(testing_predictions))

    train_incorrect = 0 
    test_incorrect = 0

    for i in range(len(train_labels_column)): 
        if training_predictions[i] != train_labels_column[i]: 
            train_incorrect += 1 
    for i in range(len(test_labels_column)): 
        if testing_predictions[i] != test_labels_column[i]: 
            test_incorrect += 1
    
    train_error_rate = train_incorrect / len(train_labels_column) 
    test_error_rate = test_incorrect / len(test_labels_column) 

    with open(command_input.metrics_out, "w") as metrics_out_file: 
        metrics_out_file.write(f"error(train): {train_error_rate}\n")
        metrics_out_file.write(f"error(test): {test_error_rate}\n")

if __name__ == "__main__":
    assert(len(sys.argv) >= 6)
    command_input = CommandInput(
        sys.argv[1],
        sys.argv[2], 
        sys.argv[3], 
        sys.argv[4], 
        sys.argv[5]
    )

    majority_classifier = MajorityVoteClassifier()
    majority_classifier.train(command_input)

    write_outputs_and_metrics(command_input, majority_classifier)





