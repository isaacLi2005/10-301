import sys


class CommandInput:
    def __init__(self, input_file, output_file):
        self.input_file = input_file 
        self.output_file = output_file 


def main(): 
    assert(len(sys.argv) >= 3) 

    command_input = CommandInput(
        sys.argv[1],
        sys.argv[2]
    )


    return 0 


if __name__ == "__main__": 
    main()

