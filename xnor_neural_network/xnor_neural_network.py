##################################################
## Author: Marton Szabo
##################################################

#      bias \
#            \
#  x1 -> O -> O -> result
#     \/    /
#  x2 -> O /
#       /
#  bias (into both)

import numpy as np

# L = total layers in the network
# sl = number of nodes in layer l

def sigmoid(x):
    return 1 / (1 + np.exp(-x));

def sigmoid_d(x):
    return x * (1 - x)

if __name__ == '__main__':

    # turn off scientific notation for printing numbers
    np.random.seed(1)
    np.set_printoptions(suppress=True)

    # we have 1 input layer, 1 hidden layer, 1 output layer, output layer has no weights
    # 3 neurons, 2 in the first layer, 1 in the output layer
    
    weightsN11 = 2 * np.random.random(size = (2)) - 1
    biasN11 = 2 * np.random.random() - 1
    weightsN12 = 2 * np.random.random(size = (2)) - 1
    biasN12 = 2 * np.random.random() - 1
    weightsN21 = 2* np.random.random(size = (2)) - 1
    biasN21 = 2 * np.random.random() - 1

    print('N11 weights: ', weightsN11)
    print('N11 bias: ', biasN11)
    print('N12 weights: ', weightsN12)
    print('N12 bias: ', biasN12)
    print('N21 weights: ', weightsN21)
    print('N21 bias: ', biasN21)

    # training input/outputs

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]

    # train neural network

    learningRate = 1 # alpha
    iterations = 5000

    # accumulators
    D11 = 0
    D12 = 0
    D21 = 0

    DB_11 = 0
    DB_12 = 0
    DB_21 = 0

    error = []

    for i in range(0, iterations):

        D11 = 0
        D12 = 0
        D21 = 0

        DB_11 = 0
        DB_12 = 0
        DB_21 = 0

        for k in range(len(inputs)):
        
            x = inputs[k]
            y = outputs[k]

            # print('training input: ', x)
            # print('training output: ', y)

            # forward propagate

            z11 = np.dot(x, weightsN11) + biasN11
            a11 = sigmoid(z11)
            z12 = np.dot(x, weightsN12) + biasN12
            a12 = sigmoid(z12)

            # print('z11: ', z11)
            # print('a11: ', a11)
            # print('z12: ', z12)
            # print('a12: ', a12)
            
            a1 = np.array([a11, a12])

            z2 = np.dot(a1, weightsN21) + biasN21
            a2  = sigmoid(z2)

            # print('z2: ', z2)
            # print('a2: ', a2)

            # backpropagate

            delta2 = a2 - y
            
            delta11 = weightsN21[0] * delta2 * sigmoid_d(a11)
            delta12 = weightsN21[1] * delta2 * sigmoid_d(a12)

            # print('delta2: ', delta2)
            # print('delta11: ', delta11)
            # print('delta12: ', delta12)

            D11 += np.dot(x, delta11)
            D12 += np.dot(x, delta12)
            D21 += a1 * delta2

            DB_11 += delta11
            DB_12 += delta12
            DB_21 += delta2

            # print('D11: ', D11)
            # print('D12: ', D12)
            # print('D21: ', D21)

            loss = 1/2 * np.power(a2 - y, 2)
            error.append(loss)

    # gradient descent

        weightsN11 -= learningRate * 1/4 * D11
        biasN11 -= learningRate * 1/4 * DB_11
        weightsN12 -= learningRate * 1/4 * D12
        biasN12 -= learningRate * 1/4 * DB_12
        weightsN21 -= learningRate * 1/4 * D21
        biasN21 -= learningRate * 1/4 * DB_21

        # print('N11 weights: ', weightsN11)
        # print('N11 bias: ', biasN11)
        # print('N12 weights: ', weightsN12)
        # print('N12 bias: ', biasN12)
        # print('N21 weights: ', weightsN21)
        # print('N21 bias: ', biasN21)

        meanError = np.mean(error[-4:])
        print('meanError: ', meanError)

        # stopping if error is small enough
        if meanError < 1e-5:
            print('Stopping after iterations: ', i)
            break;

# todos: 
# - make class for neural network
# - create functions: forward propagate, backprop, gradient descent, and predict
# - add test inputs
# - create an alternative, vectorized implementation (delta11, delta12 -> d1 for layer 1) -> new python file
# - create an alternative tensorflow implementation