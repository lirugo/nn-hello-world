import math

TRAINING_DATASET_AND = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]

TRAINING_DATASET_OR = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
]


# Main class
class NeuralNetwork():
    lr = .1  # learning rate
    bias = 1  # default value of bias
    iterations = 1000 # count of iterations
    weights = [.5, .10, .15]  # init weights can be a random numbers

    def __init__(self):
        print('Initialization -------------- ')
        print('Weights: ', self.weights)
        print('lr:', self.lr)
        print('bias:', self.bias)
        print('cout of iterations:', self.iterations)
        print('-----------------------------')

    # Activation function
    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def perceptron(self, input1, input2, output1):
        out = input1 * self.weights[0] + input2 * self.weights[1] + self.bias * self.weights[2]
        if out > 0:
            out = self.sigmoid(out)
        else:
            out = 0

        error = output1 - out
        self.weights[0] = self.weights[0] + (input1 + self.lr) * error
        self.weights[1] = self.weights[1] + (input2 + self.lr) * error
        self.weights[2] = self.weights[2] + (self.bias + self.lr) * error

    def train(self, training_set):
        for i in range(self.iterations):
            for input in training_set:
                self.perceptron(input[0], input[1], input[2])

        print('Weights: ', self.weights)

    # Get the output from NN for specific inputs
    def output(self, input1, input2):
        out = input1 * self.weights[0] + input2 * self.weights[1] + self.bias * self.weights[2]
        out = 1 if out > 0 else 0  # Apply threshold for result

        print('i1:', input1, 'i2:', input2, 'result:', out)
        return out


def main():
    nn = NeuralNetwork()

    print('Dataset for AND')
    nn.train(TRAINING_DATASET_AND)
    nn.output(1, 1)
    nn.output(1, 0)
    nn.output(0, 1)
    nn.output(0, 0)

    print('Dataset for OR')
    nn.train(TRAINING_DATASET_OR)
    nn.output(1, 1)
    nn.output(1, 0)
    nn.output(0, 1)
    nn.output(0, 0)


if __name__ == "__main__":
    main()
