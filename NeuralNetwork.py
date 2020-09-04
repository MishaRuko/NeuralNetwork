import numpy as np
import random
import tqdm

def getNetworkStructure():
    netStructure = np.load("netStructure")
    '''
    netStructure should be a 1D array containing the number of units in each layer (excluding the bias unit)
    '''
    return netStructure

def generateParameters(netStructure):
    '''
    Generates randomised weights and biases for a network
    '''

    '''
    Generate a 3D array containing (number of layers in network)-1 2D arrays
    2D arrays each contain the same number of 1D arrays as there are units in the layer that the weights are mapping to
    1D arrays contain the same number of values as there units in the layer that weights are mapping from
    '''
    weights = [ [[random.random() for _ in range(netStructure[i])] for _ in range(netStructure[i+1])] for i in range(len(netStructure)-1) ]
    
    # there are (number of layers in the network)-1 biases in the network
    biases = [random.random() for _ in range(len(netStructure)-1)]

    return weights, biases


def forwardProp(weights, biases, inputVals):
    # sigmoid activation function
    activate = lambda x: 1/(1+((np.e)**(-x)))

    # Holds all activations for the whole network:
    # [[activation values of layer 1], [activation values of layer 2], ..., [activation values of layer x]]
    activations = [inputVals]

    # Does the input layer of a network have z values (weighted sums)?
    # I think not so the input layer will have no weighet sum values
    # [[], [weighted sum for layer 1], [weighted sum of layer 2], ..., [weighted sum of layer x]]
    weightedSums = [[]]

    for i in range(len(weights)):
        # going by layers on the outside loop
        # activations.append([])

        weightedSums.append([])

        for w in weights[i]:
            # TODO check if fully vectorised implemenation is faster
            
            # Original for when you don't want weighetSums
            # If you use this then uncomment activations.append([]) above 
            # activations[i+1].append(activate(np.dot(activations[i], w)+biases[i]))

            # Each iteration add an unactivated weightedSum to weightedSums
            weightedSums[i+1].append(np.dot(activations[i], w)+biases[i])

        # Take the weighted sums for each layer, apply the activation function for each of the values, and add to activations
        activations.append(list(map(activate, weightedSums[i+1])))
        
    return activations, weightedSums


def trainNetwork(h, weights, biases, trainingExamples, function="c"):
    '''
    h - forwardProp function
    w - weight set
    b - bias set
    trainingExamples - 3D array containing training examples
    e.g. [ [[inputs for network], [outputs of network]], [[inputs of network], [outputs of network]] ]
    function - the type of cost function to be used
        Cost function list:
        c - cross-entropy
    '''

    # 2D array, contains ouputs for every training example
    predictions = [h(w, b, trainingExamples[i][0])[0][-1] for i in range(len(trainingExamples))]

    # TODO make list of activations and weighted sums for all training examples

    def getDeltas(weights, biases, weightedSums, networkOutput, desiredOutput):
        '''
        weightedSums: 2D array that holds the z values (weighted sums) for each neuron from every layer.
        networkOutput: 1D array that holds the outputs of the last neurons of a network.
        desiredOutput: similar to networkOutput but holds the desired outputs of the last neurons of a network.

        This is all for one set of input values.
        '''
        deltas = []

        if function.lower == "c":
            # Derivative of the cost function
            # there is a negative sign at the beginning because of the constant -1/m but here we take m to be 1 so it's just -1
            dJ = lambda y, output: -(y-output)/(output-(output**2))

            # Derivative of the activation function
            dSig = lambda x: (np.e**x)/((1+np.e**x)**2)

            # First get deltas for the output layer
            for i in range(len(networkOutput)):
                deltas.append(dJ(desiredOutput[i], networkOutput[i])*dSig(weightedSums[-1][i]))

            # Get deltas for rest of network
            for layer in range(len(weights)-2, 0, -1):
                for neuron in range(len(weights[layer])):
                    nueronWeights = [ weights[layer][h][neuron] for h in range()]

                    # remeber to use .insert(0, values)

    def crossEntropyCost():
        '''
        Needs predictions for every training example in array called predictions
        predictions = [h(w, b, trainingExamples[i][0])[0][-1] for i in range(len(trainingExamples))]
        '''
        result = 0
        for i in range(len(trainingExamples)):
            for j in range(len(predictions[0])):
                result += trainingExamples[i][1][j]*np.log(predictions[i][j])+(1-trainingExamples[i][1][j])*np.log(1-predictions[i][j])
    
        result = result*(-1/len(trainingExamples))
        return result

    # TODO change this to something else after done with testing
    return crossEntropyCost()
