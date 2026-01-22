import sys
import math
import numpy as np 


class DataHolder:
    def __init__(self, input_file, output_file):
        self.input_file = input_file 
        self.output_file = output_file 

        self.input_data = np.genfromtxt(
            self.input_file, 
            delimiter = "\t", 
            names = True, 
            dtype = int
        )

def get_last_column(array): 
    """
    Finds and returns the last column of a numpy array, assuming its named. 
    """
    return array[array.dtype.names[-1]]

def find_majority_vote_error(data_holder): 
    label_column = get_last_column(data_holder.input_data)

    num_0 = 0
    num_1 = 0
    
    for label in label_column: 
        if label == 0:
            num_0 += 1
        else:
            assert(label == 1)
            num_1 += 1 
    
    if num_1 >= num_0: 
        majority_vote = 1
    else:
        majority_vote = 0 

    num_input_rows = data_holder.input_data.shape[0]

    incorrect = 0 

    for i in range(num_input_rows): 
        if majority_vote != label_column[i]: 
            incorrect += 1

    majority_vote_error = incorrect / num_input_rows 

    return majority_vote_error 

def calculate_label_entropy(data_holder): 
    label_column = get_last_column(data_holder.input_data)

    num_0 = 0
    num_1 = 0
    
    for label in label_column: 
        if label == 0:
            num_0 += 1
        else:
            assert(label == 1)
            num_1 += 1 

    if num_0 == 0 or num_1 == 0: 
        return 0 

    proportion_0 = num_0 / label_column.shape[0] 
    proportion_1 = num_1 / label_column.shape[0] 

    return -1 * (proportion_0 * math.log2(proportion_0) + proportion_1 * math.log2(proportion_1))

def output_entropy_and_error(data_holder): 
    with open(data_holder.output_file, "w") as output: 
        output.write(f"entropy: {calculate_label_entropy(data_holder)}\n")
        output.write(f"error: {find_majority_vote_error(data_holder)}\n")


def main(): 
    assert(len(sys.argv) >= 3) 

    data_holder = DataHolder(
        sys.argv[1],
        sys.argv[2]
    )

    output_entropy_and_error(data_holder)

    return 0 


if __name__ == "__main__": 
    main()


"""
0.4020618556701031 error on heart
0.36 error on purchase
"""
