'''

Hi, thanks for paying attention to this work !

This code was written in collaboration with my skillful classmate Anthony Frion from IMT Atlantique, France.

There is no outstanding innovation here, but we found it quite useful to implement simple neural networks
of our own, just to be able to really understand how they work and to be able to compare it with other
structures (like the RBF networks). Also, we found it interesting to share it with anyone who would be
interested in understanding the practical aspects of machine learning ;)

So feel free to test the existing functions, maybe write some improvement, and contact me if you have
any questions or remarks :)

'''


# Imported modules

import numpy as np
import math
import time
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy as cp

########################################################

# Basic auxilary functions

def fst(couple) :
    a, b = couple
    return a

def snd(couple) :
    a, b = couple
    return b

def copy(Matrix) :
    return cp.deepcopy(Matrix)

# Gets a list of 1-element lists out of a given list L
def separate(L) :
    if len(L[0]) > 1 :
        L = [[x] for x in L[0]]
    return L

# Gets a random permutation of a list
def mix(Set) :
    copySet = copy(Set)
    mixedSet = []
    while copySet != [] :
        r = rd.randint(0,len(copySet)-1)
        mixedSet.append(copySet.pop(r))
    return mixedSet

# Auxilary function used to construct a list of regularly-disposed points in a n-dimension space
def following(previous, n) :
    result = [previous[k] for k in range(len(previous))]
    if result[0] < n :
        result[0] += 1
    else :
        result[0] = 1
        cpt = 1
        while cpt < len(previous) and result[cpt] == n :
            result[cpt] = 1
            cpt += 1
        if cpt < len(previous) :
            result[cpt] += 1
    return result

class  neuralNetWork:
    """caracterised by
    - An integer number of layers 'n'
    - An integer's list 'lengths' such that lengths[i] is the number of neurons from layer i.
    - A list of matrices 'weights' (n-1 matrices). Each matrix contains the weights of all connexions between 2 consecutive layers
    The size of weights[i] is lengths[i+1]*lengths[i].
    - A list 'biaises' of n-1 biaises lists, each of which corresponds to a layer. The input layer has no biaises.
    - A list of layers 'layers'. Each layer is a list of real numbers of length lengths[i].
    - A learning rate 'etha' which is a float number.
    """

    # The class constructor. The only arguments are the dimensions 'lengths' and the learning rate etha,
    # as the weights and biaises are initialised randomly
    def __init__(self, lengths, etha):
        self.numberOfLayers = len(lengths)
        self.lengths = lengths
        initialBiaises = []
        for c in range(1, self.numberOfLayers):
            B =  np.zeros((self.lengths[c],1))
            for i in range(len(B)):
                B[i][0] = rd.random()
            initialBiaises.append(B)
        self.biaises = initialBiaises
        initialWeights = []
        for c in range(self.numberOfLayers-1):
            W = np.zeros((lengths[c+1], lengths[c]))
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] = rd.random()
            initialWeights.append(W)
        self.weights = initialWeights
        layers = []
        for i in range(self.numberOfLayers):
            couche = np.zeros((self.lengths[i],1))
            layers.append(couche)
        self.layers = layers
        self.etha = etha

    # Computes an output from the network to a given inuput
    def propagate(self,input):
        #input is a vector containing input data to the neural network
        self.layers[0] = input

        for i in range(1,self.numberOfLayers-1):
            self.layers[i] = activationFonction(np.dot(self.weights[i-1],self.layers[i-1])+self.biaises[i-1])
        
        self.layers[self.numberOfLayers-1] = np.dot(self.weights[self.numberOfLayers-2],self.layers[self.numberOfLayers-2])+self.biaises[self.numberOfLayers-2]
        return self.layers[self.numberOfLayers-1]

    def iterationOn(self, input, expectedOutput):
        # Gets an input which it calculates the output and compares it to the expected output.
        # Then readjusts the weights and biaises accordingly.
        # This is an online-training methods as the computations are done input by input.
        output = self.propagate(input)
        expectedOutput = separate(expectedOutput)
        error = expectedOutput - output
        self.backPropagate(error)


    def xhi(self, couche):
        return np.dot(self.weights[couche-1], self.layers[couche-1]) + self.biaises[couche-1]

    # Backpropagation algorithm -> modifies the weights and biaises according to the error
    def backPropagate(self, error):
        L = self.numberOfLayers
        deltaC = error
        self.modifyWeights(L-1, deltaC)
        self.biaises[L-2] += self.etha * deltaC
        for c in range(L-2,0,-1):
            deltaC = np.transpose(self.weights[c]).dot(deltaC) * tanhprim(self.xhi(c))
            self.modifyWeights(c,deltaC)
            self.biaises[c-1] += self.etha * deltaC

    # Modifies the weights from the network
    def modifyWeights(self, c, deltaC):
        for i in range(len(self.weights[c-1])):
            for j in range(len(self.weights[c-1][0])):
                self.weights[c-1][i][j] += self.etha*deltaC[i]*self.layers[c-1][j]
    
    # Gets the loss of the vector, which is the sum of squared errors when approximated a function on a validation set
    def getLoss(self, validationSet, function) :
        X = [self.propagate(np.transpose(np.array([x]))) for x in validationSet]
        Y = [np.array([function(x)]) for x in validationSet]
        loss = sum(sum([(X[k] - Y[k])**2 for k in range(len(X))]))
        loss = sum(loss)
        return loss
            
    '''
    First training function. The arguments are :
    - a function to approximate
    - a training set which will be used for training the networks with backpropagations
    - a validation set which is not used for training but for calculating the loss
    - a batch_size, which is the number of times that the backpropagation will occur on the test set
    - a number of iterations of the whole process
    - an optional parameter that can stop the training if the loss increases between two iterations
    '''
    def train(self, function, trainingSet, validationSet, batch_size, nbIterations=5, stop=False):
        print("Training with etha = " + str(self.etha))
        best_loss = self.getLoss(validationSet, function)
        new_loss = best_loss
        best_biaises = copy(self.biaises)
        best_weights = copy(self.weights)
        for i in range(nbIterations) :
            for k in range(batch_size) :
                #trainingSet = mix(trainingSet)
                for x in trainingSet :
                    input = np.transpose(np.array([x]))
                    expectedOutput = np.array([function(x)])
                    self.iterationOn(input, expectedOutput)
            new_loss = self.getLoss(validationSet, function)
            print("Loss is " +  str(new_loss))
            if new_loss < best_loss :
                best_weights = copy((self.weights))
                best_biaises = copy((self.biaises))
                best_loss = new_loss
            elif stop :
                break
        self.weights = best_weights
        self.biaises = best_biaises
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
        return self

    '''
    A more advanced training function.
    This time the number of iterations is not fixed, but the process will stop if some conditions
    linked with time1 and time2 are verified. The training will be prolonged if the loss decreases
    fast enough, but it will necessarily stop at one point.
    The parameter divideEthaBy is the number by which the learning rate will be divided at each iteration.
    The parameter selective training, if activated, will stop the network from training on the already
    learnt points (which are the ones which error is at least 5 times smaller than the average one).
    '''
    def train2(self, function, trainingSet, validationSet, batch_size, time1=20, time2=30, divideEthaBy=1, selectiveTraining=True) :
        old_loss = self.getLoss(validationSet, function)
        new_loss = old_loss*0.95
        best_loss = old_loss
        print("Loss is " +  str(self.getLoss(validationSet, function)))
        best_biaises = copy(self.biaises)
        best_weights = copy(self.weights)
        t0 = time.time()
        t1 = t0
        while time.time() - t0 < time1 and new_loss <= old_loss*(3 - min(2.001, (time.time() - t1)*6/time2)) :
            #trainingSet = mix(trainingSet)
            for k in range(batch_size) :
                for x in trainingSet :
                    input = np.transpose(np.array([x]))
                    expectedOutput = np.array([function(x)])
                    error = expectedOutput - self.propagate(input)
                    if not selective_training or sum(sum([x**2 for x in error])) >= new_loss / (5*len(validationSet)) :
                        self.iterationOn(input, expectedOutput)
            old_loss = new_loss
            new_loss = self.getLoss(validationSet, function)
            print("Loss is " +  str(self.getLoss(validationSet, function)))
            if new_loss < best_loss :
                t1 = time.time()
                best_biaises = copy(self.biaises)
                best_weights = copy(self.weights)
                if new_loss <= 0.98*best_loss :
                    t0 = time.time()
                best_loss = new_loss
            if new_loss < best_loss :
                t1 = time.time()
                best_weights = copy(self.weights)
                best_biaises = copy(self.biaises)
                if new_loss <= 0.98*best_loss :
                    t0 = time.time()
                best_loss = new_loss
        self.weights = best_weights
        self.biaises = best_biaises
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
        return self
    
    '''
    Similar to train but this time there is no given function among the arguments,
    and the training set and validation set are not only lists of inputs, instead they are lists
    that each contain an input and the corresponding expected output
    '''
    def train3(self, trainingSet, validationSet, batch_size, nbIterations=5, stop=False):
        best_loss = self.getLoss3(validationSet)
        new_loss = best_loss
        best_biaises = copy(self.biaises)
        best_weights = copy(self.weights)
        t0 = time.time()
        for k in range(nbIterations) :
            for k in range(batch_size) :
                for x in trainingSet :
                    input = fst(x)
                    expectedOutput = snd(x)
                    self.iterationOn(input, expectedOutput)
            new_loss = self.getLoss3(validationSet)
            print("Loss is " +  str(self.getLoss3(validationSet)))
            if new_loss < best_loss :
                best_weights = copy((self.weights))
                best_biaises = copy((self.biaises))
                best_loss = new_loss
            elif stop :
                break
        self.weights = best_weights
        self.biaises = best_biaises
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss3(validationSet)))
        return self
    
    '''
    Similar to train but this time there is no given function among the arguments,
    and the training set and validation set are not only lists of inputs, instead they are lists
    that each contain an input and the corresponding expected output
    '''
    def train4(self, trainingSet, validationSet, batch_size, temps1=20, temps2=30, ethaConstant=True, divideEthaBy=1) :
        self.etha /= divideEthaBy
        old_loss = self.getLoss3(validationSet)
        new_loss = old_loss*0.95
        best_loss = old_loss
        print("Loss is " +  str(self.getLoss3(validationSet)))
        best_biaises = copy(self.biaises)
        best_weights = copy(self.weights)
        t0 = time.time()
        t1 = t0
        while time.time() - t0 < temps1 and new_loss <= old_loss*(3 - min(2.001, (time.time() - t1)*6/temps2)) :
            if not ethaConstant :
                print("current etha is " + str(self.etha))
            self.etha /= divideEthaBy
            for k in range(batch_size) :
                #trainingSet = mix(trainingSet)
                for x in trainingSet :
                    input = fst(x)
                    expectedOutput = snd(x)
                    self.iterationOn(input, expectedOutput)
            old_loss = new_loss
            new_loss = self.getLoss3(validationSet)
            print("Loss is " +  str(self.getLoss3(validationSet)))
            
            if not ethaConstant :
                """if new_loss <= best_loss and new_loss/old_loss >= .9 :
                    self.etha *= np.sqrt(2.02 - new_loss/best_loss)
                elif min(abs(new_loss/best_loss), abs(best_loss/new_loss)) <= .95 :
                    self.etha *= min(abs(new_loss/best_loss), abs(best_loss/new_loss))
                """
            if new_loss < best_loss :
                t1 = time.time()
                best_biaises = copy(self.biaises)
                best_weights = copy(self.weights)
                if new_loss <= 0.99*best_loss :
                    t0 = time.time()
                best_loss = new_loss
        self.weights, self.biaises = best_weights, best_biaises
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss3(validationSet)))
        return self
           
    # Gets the loss of the network out of a validation set that contains a list of couples
    # which first element is an output and the second element is the expected corresponding output
    def getLoss3(self, validationSet) :
        X = [self.propagate(fst(x)) for x in validationSet]
        Y = [snd(x) for x in validationSet]
        loss = sum(sum([(X[k] - Y[k])**2 for k in range(len(X))]))
        return loss

# The activation function that will be used to compute the content of each neuron.
# This implementation has chosen the hyperbolic tangent.
def activationFonction(x): 
    return np.tanh(x)

# The derivative of hyperbolic tangent
def tanhprim(x):
    return 1 - np.tanh(x)**2

# A list of test function.
# You can choose any one of those by just uncommenting the corresponding line.
# You then have to change the 'nbInputs' and 'nbOutputs' accordingly (and perhaps the learning rates range)
def test_func(x):
    return [4*(x[0]-1/4)*(x[0]-.4)*(x[0]-2/3)]
    #return [(math.cos(2*np.pi*x[0]) +1)/2]
    #return [np.sinc(x[0]*3)]
    #return [x[0] + x[1]]
    #return [np.exp(x[0]) / 3]
    #return [1,0] if x[0] > 0.5 else [0.5,0.5]
    
# Global parameters that corresponds to the structure of the network.
# Choose them according to the function you want to approximate.
nbInputs = 1
nbOutputs = 1


# Plots the test function and its approximation by a neural network. Only relevant for one-input-one-output functions.
def plotResult(neuNet, function) :
    X = [k/1000 for k in range(1000)]
    Y = [neuNet.propagate(x)[0][0] for x in X]
    Z = [function(np.transpose(np.array([x]))) for x in X]
    plt.plot(X,Y)
    plt.plot(X,Z)
    plt.show()

# Creates training, validation and test sets of points that are regularly placed between 0 and 1 in all
# the dimensions. These points are randomly placed in the 3 sets.
def createSets(nbPoints, nbDimensions) :
    n = int(nbPoints ** (1/nbDimensions))
    X = [[0 for k in range(nbDimensions)] for i in range(nbPoints)]
    X[0] = [0 for k in range(nbDimensions)]
    for i in range(1, nbPoints) :
        X[i] = following(X[i-1], n)
    for i in range(len(X)) :
        for k in range(nbDimensions) :
            X[i][k] = ((.5 + X[i][k]) / n)
    trainingSet = []
    validationSet = []
    for k in range(int(len(X)*0.70)) :
        r = rd.randint(0,len(X)-1)
        trainingSet.append(X.pop(r))
    for k in range(int(len(X)/2)) :
        r = rd.randint(0, len(X)-1)
        validationSet.append(X.pop(r))
    testSet = X
    return trainingSet, validationSet, testSet

# Initiates the graphical animation.
def initiate_graphics() :
    global figure
    global axes
    figure = plt.figure()
    axes = plt.axes()
    axes.set_ylim(-.5, 0.8)
    X = [k/1000 for k in range(1000)]
    plt.plot(X, [test_func(np.transpose(np.array([x]))) for x in X], "blue")
X = []
Y = []   
new_loss = 1000 

# Updates the graphics to keep the animation going.
def update_graphics(i) :
    global neuNet
    global figure
    global X
    global Y
    global new_loss
    for x in trainingSet :
        input = np.transpose(np.array([x]))
        expectedOutput = np.array([test_func(x)])
        error = expectedOutput - neuNet.propagate(input)
        if sum(sum([x**2 for x in error])) >= new_loss / (6*len(validationSet)) :
            neuNet.iterationOn(input, expectedOutput)
    plt.plot(X, Y, color=(1, 1, 1))
    if not leaveTrace :
        for k in range(-10,10) :
            plt.plot(X, [y + k*.0002 for y in Y], color=(1,1,1))
        plt.plot(X, [test_func(np.transpose(np.array([x]))) for x in X], "blue")
    X = [k/1000 for k in range(1000)]
    Y = [neuNet.propagate(x)[0][0] for x in X]
    plt.plot(X,Y, "red")
    new_loss = neuNet.getLoss(validationSet, test_func)
    print("Loss is " +  str(new_loss))
    
# The totl number of points that you will use to train your neural network.
# You usually need many points when the number of dimensions is higher.
nbPoints = 1000

# Creation of the training, validation and test sets.
trainingSet, validationSet, testSet = createSets(nbPoints, nbInputs)

# Make this parameter true if you want to see the previous iterations of the training in the animation.
leaveTrace = False

'''
A procedure that trains a neural network to approximate the global-parametered function 'test_func'.
It first uses the first training function with different learning rates to try to get one that gives good results.
Then it keeps on the training with this learning rate and slowly decreases it to get more precision.
You can choose the range of ethas on which you will make the first phase of training with the first three parameters.
We recommand that you choose etha from 0.01 to 0.20 for one-dimensional functions and smaller ethas for higher dimensions.
'''
def procedure(startingEtha, stepEtha, numberEthas, plot = False) :
    neuNets = []
    lossCand = 1000
    for i in range(numberEthas) :
            etha = startingEtha + stepEtha*i
            neuNets.append(neuralNetWork([nbInputs, 10, nbOutputs], etha))
            neuNets[i].train(test_func, trainingSet, validationSet, 2, 10, True)
            loss = neuNets[i].getLoss(validationSet, test_func)
            if loss < lossCand :
                lossCand = loss
                neuNetCand = neuNets[i]
                ethacand = etha
                print("New best loss is " + str(lossCand))
    print("Further training for the cand which etha is " + str(ethacand) + " and loss is " + str(lossCand))
    print("Verif : loss from neuNetCand is " + str(neuNetCand.getLoss(testSet, test_func)))
    neuNetCand.train2(test_func, trainingSet, validationSet, 1, 30, 30, True, 1.01)
    neuNetCand.train2(test_func, trainingSet, validationSet, 1, 60, 60, True, 1.02)
    print(neuNetCand.getLoss(testSet, test_func))
    if plot :
        plotResult(neuNetCand, test_func, trainingSet, validationSet, testSet)
      
# Creates an animation showing how the neural network approximates the test function over the training.
# Note that the training process is very basic here as it is done with a constant learning rate : 
# you could want to animate a more complex training (like the one of the previous procedure)
def drawAnimation(etha, nbImages) :
    global figure
    global neuNet
    initiate_graphics()
    neuNet = neuralNetWork([nbInputs, 10, nbOutputs], etha)
    anim = animation.FuncAnimation(figure, update_graphics, frames=nbImages, interval=100)
    plt.show()
    

if __name__ == "__main__":
    #procedure(0.01, 0.02, 10, False, True)
    drawAnimation(0.1, 10)