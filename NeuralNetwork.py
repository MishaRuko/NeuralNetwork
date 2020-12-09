import numpy as np
import random
from tqdm import tqdm


def getNetworkStructure():
    """
    netStructure should be a 1D array containing the number of units in each layer (excluding the bias unit)
    """
    netStructure = np.load("netStructure")

    return netStructure


def generateParameters(netStructure):
    """
    Generates randomised weights and biases for a network
    Returns weights and biases
    weights[weightSet][nextNeuron][Neuron]
    biases[weightSet]
    """

    """
    Generate a 3D array containing (number of layers in network)-1 2D arrays
    2D arrays each contain the same number of 1D arrays as there are units in the layer that the weights are mapping to
    1D arrays contain the same number of values as there units in the layer that weights are mapping from
    weights[layer mapping from][neuron mapping to][neuron mapping from]
    """

    weights = []
    for weightSet in range(len(netStructure) - 1):
        weights.append([])
        for nextNeuron in range(netStructure[weightSet + 1]):
            weights[weightSet].append([])
            for neuron in range(netStructure[weightSet]):
                if random.randint(0, 1) % 2 == 0:
                    weights[weightSet][nextNeuron].append(random.random())
                else:
                    weights[weightSet][nextNeuron].append(-random.random())

    biases = []
    for weightSet in range(len(netStructure) - 1):
        if random.randint(0, 1) % 2 == 0:
            biases.append(random.random())
        else:
            biases.append(-random.random())

    return weights, biases


def forwardProp(weights, biases, networkInputs, layerActivationFuncs):
    """
    layerActivationFuncs - array containing the type of activation function to be used by each layer\n
    len(layerActivationFuncs) == len(weights)\n
    Valid layerActivationFuncs values:\n
    "leakyReLU" - leaky ReLU activation function - max(0.01x, x)
    "sigmoid" - sigmoid/logistic activation function\n

    Returns activations and weightedSums
    activations[layer][neuron]
    weightedSums[layer][neuron]
    weightedSums[0] == []
    """

    if len(layerActivationFuncs) != len(weights):
        raise ValueError("layerActivationFuncs has an invalid size: {}, size should be: {}".format(
            len(layerActivationFuncs), len(weights)))

    # Activation functions
    # WARNING changed here
    def sigmoid(x):
        if x > 22:
            return 0.999999
        elif x < -22:
            return 1e-10
        return 1 / (1 + (np.e ** (-x)))

    def leakyReLU(x):
        return max(0.01 * x, 0.5 * x)

    # Holds all activations for the whole network:
    # [[activation values of layer 1], [activation values of layer 2], ..., [activation values of layer x]]
    activations = [networkInputs]

    # [[], [weighted sum for layer 1], [weighted sum of layer 2], ..., [weighted sum of layer x]]
    weightedSums = [[]]

    for weightSet in range(len(weights)):
        weightedSums.append([])
        for nextNeuron in weights[weightSet]:
            # TODO check if fully vectorised implementation is faster
            # Each iteration add an unactivated weightedSum to weightedSums
            weightedSums[weightSet + 1].append(np.dot(activations[weightSet], nextNeuron) + biases[weightSet])

        # Take the weighted sums for each layer, apply the correct activation function for each of the values,
        # and add to activations
        if layerActivationFuncs[weightSet] == "sigmoid":
            activations.append(list(map(sigmoid, weightedSums[weightSet + 1])))
        elif layerActivationFuncs[weightSet] == "leakyReLU":
            activations.append(list(map(leakyReLU, weightedSums[weightSet + 1])))
        else:
            raise ValueError("Invalid activation function {}".format(layerActivationFuncs[weightSet]))

    return activations, weightedSums


def trainNetwork(h, weights, biases, layerActivationFuncs, trainingExamples, alpha, lambd, func="c", iters=10):
    """
    h - forwardProp function\n
    weights - weight set\n
    biases - bias set\n
    layerActivationFuncs - array containing the type of activation function to be used by each layer\n
    len(layerActivationFuncs) == len(weights)\n
    Valid layerActivationFuncs values:\n
    "leakyReLU" - leaky ReLU activation function - max(0.01x, x)\n
    "sigmoid" - sigmoid/logistic activation function\n
    trainingExamples - 3D array containing training examples\n
    e.g. [ [[inputs for network], [outputs of network]], [[inputs of network], [outputs of network]] ]\n
    func - the type of cost function to be used\n
        Cost function list:\n
        c - cross-entropy\n
    iters - number of iterations for gradient descent\n

    Returns trained weights and biases
    """

    if len(layerActivationFuncs) != len(weights):
        raise ValueError("layerActivationFuncs has an invalid size: {}, size should be: {}".format(
            len(layerActivationFuncs), len(weights)))

    def crossEntropyCost(networkOutput):
        """Returns cross entropy cost of network"""

        def squaredSumWeights():
            return sum([[[weights[i][j][k]**2 for k in range(len(weights[i][j]))]
                         for j in range(len(weights[i]))]
                        for i in range(len(weights))])

        result = 0
        # for every training example
        for trainingExample in range(len(trainingExamples)):
            # for the number of outputs neurons there are
            for output in range(len(networkOutput)):
                result += (trainingExamples[trainingExample][1][output]
                           * np.log(networkOutput[output])
                           + (1 - trainingExamples[trainingExample][1][output])
                           * np.log(1 - networkOutput[output])
                           + (lambd/(2*len(trainingExamples)))*squaredSumWeights())        # WARNING

        result = result * (-1 / len(trainingExamples))
        return result

    def getDeltas(weightedSums, networkOutput, desiredOutput):
        """
        weightedSums: 2D array that holds the z values (weighted sums) for each neuron from every layer.
        networkOutput: 1D array that holds the outputs of the last neurons of a network.
        desiredOutput: similar to networkOutput but holds the desired outputs of the last neurons of a network.

        This is all for one set of input values.

        Returns 2D array
        deltas[deltaSet][neuron]
        """

        deltas = []

        def dCrossEntropy(y, yHat):
            maxVal = 100000
            if y == yHat:
                return 0
            elif y == 0 and yHat == 1:
                return maxVal
            elif y == 1 and yHat == 0:
                return -maxVal
            else:
                return -(y - yHat) / (yHat - (yHat ** 2))
        # def dCrossEntropy(y, yHat):
        #     return -(y - yHat) / (yHat - (yHat ** 2))

        def dSig(x):
            if abs(x) > 22:
                return 1e-10
            return (np.e ** x) / ((np.e ** x) + 1) ** 2

        def dLeakyReLU(x):
            return 0.5 if x >= 0 else 0.01

        if func == "c":
            if layerActivationFuncs[-1] == "sigmoid":
                deltas.append(
                    [dCrossEntropy(desiredOutput[neuron], networkOutput[neuron]) * dSig(weightedSums[-1][neuron])
                     for neuron in range(len(networkOutput))])
            elif layerActivationFuncs[-1] == "leakyReLU":
                deltas.append(
                    [dCrossEntropy(desiredOutput[neuron], networkOutput[neuron]) * dLeakyReLU(weightedSums[-1][neuron])
                     for neuron in range(len(networkOutput))])
            else:
                raise ValueError("Invalid activation function {}".format(layerActivationFuncs[-1]))

            for weightSet in range(len(weights) - 1, 0, -1):
                deltas.insert(0, [])
                for neuron in range(len(weights[weightSet][0])):
                    neuronWeights = [weights[weightSet][nextNeuron][neuron]
                                     for nextNeuron in range(len(weights[weightSet]))]
                    neuronDelta = np.dot(deltas[1], neuronWeights)
                    if layerActivationFuncs[weightSet] == "sigmoid":
                        neuronDelta = neuronDelta * dSig(weightedSums[weightSet][neuron])
                    elif layerActivationFuncs[weightSet] == "leakyReLU":
                        neuronDelta = neuronDelta * dLeakyReLU(weightedSums[weightSet][neuron])

                    deltas[0].append(neuronDelta)

        return deltas

    def gradientDescent():
        for _ in tqdm(range(iters)):
            for trainingExample in range(len(trainingExamples)):
                activations, weightedSums = h(weights, biases,
                                              trainingExamples[trainingExample][0], layerActivationFuncs)

                deltas = getDeltas(weightedSums, activations[-1],
                                   trainingExamples[trainingExample][1])

                for weightSet in range(len(weights)):
                    for nextNeuron in range(len(weights[weightSet])):
                        for neuron in range(len(weights[weightSet][nextNeuron])):
                            weights[weightSet][nextNeuron][neuron] -= (alpha
                                                                       * deltas[weightSet][nextNeuron]
                                                                       * activations[weightSet][neuron]
                                                                       + ((lambd/len(trainingExamples))
                                                                          * weights[weightSet][nextNeuron][neuron]))

                    biases[weightSet] -= alpha * np.sum(deltas[weightSet])

        return weights, biases
    
