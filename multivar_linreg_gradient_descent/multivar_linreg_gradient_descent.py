##################################################
## Author: Marton Szabo
##################################################

import numpy as np

if __name__ == '__main__':

    # turn off scientific notation for printing numbers
    np.set_printoptions(suppress=True)

    # parameters of dragons
    #
    # t1: number of heads 
    # t2: age
    # t3: weight
    #
    # predict how many hit points the dragons have

    baseHp = 100 # every dragon has at least 100 hp + the computed

    minNumberOfHeads = 1
    maxNumberOfHeads = 7

    minAge = 20 # years, hatchling
    maxAge = 3000 # years, oldest wyrm ever

    minWeight = 1000 # skinny dragon
    maxWeight = 8000 # fat dragon

    trainingDataSize = 100;

    epsilon = 0.000001

    # generate training dataset
    heads = np.random.randint(low=minNumberOfHeads, high=maxNumberOfHeads+1, size=trainingDataSize) # so many heads
    ages = np.random.randint(low=minAge, high=maxAge+1, size=trainingDataSize) # years old
    weights = np.random.randint(low=minWeight, high=maxWeight+1, size=trainingDataSize) # kg

    print ('heads: ', heads)
    print ('ages: ', ages)
    print ('weigths: ', weights)

    # the more heads the more dangerous, however weight does not help, age makes linear difference 
    # these params are the ones to find out by the gradient descent
    # they are just used to generate the training outputs
    params = np.array([baseHp, 250, 1, 0.7]) 

    print ('Base theta parameters:')
    print (params)

    ones = np.ones(trainingDataSize)

    # feature scaling 
    # without feature scaling it will never converge
    # computing scaled values based on standard deviation
    heads = (heads - np.mean(heads)) / (maxNumberOfHeads - minNumberOfHeads)
    ages = (ages - np.mean(ages)) / (maxAge - minAge)
    weights = (weights - np.mean(weights)) / (maxWeight - minWeight)

    # prepending ones to be able to multiply matrixes easily
    trainingData = np.column_stack((ones, heads, ages, weights))

    print ('Generated training data matrix:')
    print (trainingData)

    trainingOutputs = np.dot(trainingData, params);

    print ('Generated training outputs vector:')
    print (trainingOutputs)

    # learn parameters from training input

    learnedParams = np.random.random(size = 4)

    print ('Generated random parameters for learning:')
    print (learnedParams)

    alpha = 0.3 # learningRate
    iterations = 10000 # number of iterations

    for i in range(iterations):

        # print ('=========================================')
        # print ('iteration ', i)

        prediction = np.dot(trainingData, learnedParams)
        # print ('prediction: ', prediction)

        error = prediction - trainingOutputs

        # print ('error: ', error)

        delta = 1 / trainingDataSize * np.dot(error, trainingData);

        # print ('delta:')
        # print (delta) 

        adjustment = alpha * delta

        # print ('adjustmnent:')
        # print (adjustment) 

        learnedParams = learnedParams - adjustment

        # print ('new params:')
        # print (learnedParams) 

        # todo: make this check better
        check = np.absolute(adjustment)
        if(check[0]<epsilon and check[1]<epsilon and check[2]<epsilon and check[3]<epsilon):
            print('Converged in iterations: ', i)
            break


testDragon = np.array([1, 6, 300, 3000])

# should be around 4000
hp = np.dot(testDragon.T, learnedParams)
print ('Test dragon has hp: ', hp)

# todos: 
# implement stochastic gradient descent
# plot gradient descent -> matplotlib
# optimizations