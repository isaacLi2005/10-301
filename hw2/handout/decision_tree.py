import argparse
import numpy as np 

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
