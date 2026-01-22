import argparse
import numpy as np 
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, attr, v):
        self.attribute = attr 
        self.left = None 
        self.right = None 
        self.vote = v

class DataHolder: 
    def __init__(self, args): 
        self.train_input_data = np.genfromtxt(
            args.train_input, 
            delimiter = "\t", 
            names = True, 
            dtype = int
        )

        self.test_input_data = np.genfromtxt(
            args.test_input, 
            delimiter = "\t", 
            names = True, 
            dtype = int
        )

def calculate_column_entropy(data, column_name): 
    column = data[column_name]
    num_rows = column.shape[0]

    num_0, num_1 = count_ones_and_zeroes(data, column_name)
    
    entropy = 0

    proportion_0 = num_0 / num_rows
    entropy += proportion_0 * math.log2(proportion_0)

    proportion_1 = num_1 / num_rows 
    entropy += proportion_1 * math.log2(proportion_1)
    
    entropy *= -1 

    return entropy

def split(data, column_name): 
    X_0 = data[data[column_name] == 0]
    X_1 = data[data[column_name] == 1]

    return X_0, X_1

def count_ones_and_zeroes(data, column_name): 
    num_0 = 0 
    num_1 = 0 

    data_column = data[column_name] 
    for elem in data_column: 
        assert(elem == 0 or elem == 1)
        if elem == 0: 
            num_0 += 1 
        else:
            num_1 += 1
    
    return num_0, num_1


def calculate_mutual_information(data, label_column_name, conditional_column_name): 
    HY = calculate_column_entropy(data, label_column_name) 

    X_0, X_1 = split(data, conditional_column_name)

    conditional_num_0, conditional_num_1 = count_ones_and_zeroes(data, conditional_column_name)
    total = data.shape[0]

    HYX0 = (conditional_num_0 / total) * calculate_column_entropy(X_0, label_column_name) 
    HYX1 = (conditional_num_1 / total) * calculate_column_entropy(X_1, label_column_name)

    HYX = HYX0 + HYX1 

    mutual_information = HY - HYX 

    return mutual_information 


def print_tree(node):
    pass

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    #Here's an example of how to use argparse
    print_out = args.print_out

    data_holder = DataHolder(args) 

    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)
