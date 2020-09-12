import numpy as np
import random
from tqdm import tqdm

def getNetworkStructure():
    netStructure = np.load("netStructure")
    '''
    netStructure should be a 1D array containing the number of units in each layer (excluding the bias unit)
    '''
    return netStructure

def generateParameters(netStructure):
    '''
    Generates randomised weights and biases for a network
    Returns weights and biases 
    weights[weightSet][nextNeuron][Neuron]
    biases[weightSet]
    '''
    
    '''
    Generate a 3D array containing (number of layers in network)-1 2D arrays
    2D arrays each contain the same number of 1D arrays as there are units in the layer that the weights are mapping to
    1D arrays contain the same number of values as there units in the layer that weights are mapping from
    weights[layer mapping from][neuron mapping to][neuron mapping from]
    '''
    weights = [ [[random.random() for _ in range(netStructure[i])] for _ in range(netStructure[i+1])] for i in range(len(netStructure)-1) ]
    
    # there are (number of layers in the network)-1 biases in the network
    biases = [random.random() for _ in range(len(netStructure)-1)]

    return weights, biases


def forwardProp(weights, biases, inputVals):
    """
    Returns activations and weightedSums
    activations[layer][neuron]
    weightedSums[layer][neuron]
    """
    # sigmoid activation function
    activate = lambda x: 1/(1+((np.e)**(-x)))

    # Holds all activations for the whole network:
    # [[activation values of layer 1], [activation values of layer 2], ..., [activation values of layer x]]
    activations = [inputVals]

    # [[], [weighted sum for layer 1], [weighted sum of layer 2], ..., [weighted sum of layer x]]
    weightedSums = [[]]

    for i in range(len(weights)):
        # going by layers on the outside loop

        weightedSums.append([])

        for w in weights[i]:
            # TODO check if fully vectorised implemenation is faster
            
            # Each iteration add an unactivated weightedSum to weightedSums
            weightedSums[i+1].append(np.dot(activations[i], w)+biases[i])

        # Take the weighted sums for each layer, apply the activation function for each of the values, and add to activations
        activations.append(list(map(activate, weightedSums[i+1])))
        
    return activations, weightedSums


def trainNetwork(h, weights, biases, trainingExamples, alpha, func="c", iters=10):
    '''
    h - forwardProp function
    w - weight set
    b - bias set
    trainingExamples - 3D array containing training examples
    e.g. [ [[inputs for network], [outputs of network]], [[inputs of network], [outputs of network]] ]
    function - the type of cost function to be used
        Cost function list:
        c - cross-entropy
    iters - number of iterations for gradient descent
    
    Returns trained weights and biases
    '''
    
    def crossEntropyCost(allActivations):
        """
        Doesn't take params
        """
        result = 0
        # for every training example
        for i in range(len(trainingExamples)):
            # for the number of outputs neurons there are
            for j in range(len(allActivations[0][-1])):
                result += trainingExamples[i][1][j]*np.log(allActivations[i][-1][j])+(1-trainingExamples[i][1][j])*np.log(1-allActivations[i][-1][j])
    
        result = result*(-1/len(trainingExamples))
        return result

    def getDeltas(weights, biases, weightedSums, networkOutput, desiredOutput):
        '''
        weightedSums: 2D array that holds the z values (weighted sums) for each neuron from every layer.
        networkOutput: 1D array that holds the outputs of the last neurons of a network.
        desiredOutput: similar to networkOutput but holds the desired outputs of the last neurons of a network.

        This is all for one set of input values.

        Returns 2D array
        deltas[deltaSet][neuron]
        '''

        # first entry in array are the deltas for layer 2
        deltas = []
        # if function is c for cross-entropy cost function
        if func.lower() == "c":
            # Derivative of the cost function
            # there is a negative sign at the beginning because of the constant -1/m but here we take m to be 1 so it's just -1
            dJ = lambda y, output: -(y-output)/(output-(output**2))

            # Derivative of the activation function
            dSig = lambda x: (np.e**x)/((1+np.e**x)**2)

            # First get deltas for the output layer
            deltas.append([dJ(desiredOutput[i], networkOutput[i])*dSig(weightedSums[-1][i]) for i in range(len(networkOutput))])

            # Get deltas for rest of network
            # For every layer except the first (can't get deltas for that one) and last one (already got those)
            for weightSet in range(len(weights)-1, 0, -1):
                deltas.insert(0, [])
                # for the number of neurons in layer "weightSet"
                for neuron in range(len(weights[weightSet])):
                    neuronWeights = [ weights[weightSet][nextNeuron][neuron] for nextNeuron in range(len(weights[weightSet]))]
                    
                    neuronDelta = np.dot(neuronWeights, deltas[1]) * dSig(weightedSums[weightSet][neuron])
                    deltas[0].append(neuronDelta)

        return deltas

    def gradientDescent(alpha, iters):

        # WARNING weights and biases arrays are not made global, but the code works, might have to fix that

        # change the weights a bunch of times
        for _ in tqdm(range(iters)):
            # For every training example
            for trainingExample in range(len(trainingExamples)):
                activations, weightedSums = h(weights, biases, trainingExamples[trainingExample][0])
                deltas = getDeltas(weights, biases, weightedSums, activations[-1], trainingExamples[trainingExample][1])
                # For every weight
                for weightSet in range(len(weights)):
                    for nextNeuron in range(len(weights[weightSet])):
                        for weight in range(len(weights[weightSet][nextNeuron])):
                            # probably works
                            weights[weightSet][nextNeuron][weight] -= alpha*deltas[weightSet][nextNeuron]*activations[weightSet][weight]

                    biases[weightSet] -= alpha*np.sum(deltas[weightSet])

        return weights, biases


    return gradientDescent(alpha, iters)
