'''
Hi, thanks for paying attention to this work !

This code was written in collaboration with my skillful classmate Anthony Frion from IMT Atlantique, France.

After implementing the MLP neural networks structure (which you can also found on my Github), 
we wanted to test one of the main other structures, which is the one based on radial basis functions (RBF).

Althrough it is slower and often less effective at aprroximating than the MLP, we found a few examples
on which it performed quite good : as an example, we obtained good results on approximating periodic
functions like the cosine while the MLP structure tends to struggle on this type of problems.

Feel free to test the existing functions, maybe write some improvement, and contact me if you have
any questions or remarks :)

Thibaut
'''

############################################################################################

# Global parameters that you could want to modify.
# If you don't change anything, then you will see the animated approximation of a polynomial function.

# The total number of points that you will use to train your neural network.
# You usually need many points when the number of dimensions is higher.
nbPoints = 200

# Make this parameter true if you want to see the previous iterations of the training in the animation.
leaveTrace = False

# If activated, the selective training will stop the network from training on the already
# learnt points (which are the ones which error is at least 5 times smaller than the average one)
selective_training = True

# A list of test functions.
# You can choose any one of those by just uncommenting the corresponding line.
# You then have to change the 'nbInputs' and 'nbOutputs' accordingly (and perhaps the learning rates range)
def f(x):
    #return [(math.cos(3*np.pi*x[0]) +1)/2]
    return [abs(x[0] - 1/3) + abs(x[0] - 2/3)]
    #return [0] if x[0] < 0.5 else [1]
    #return [np.sinc(x[0]*4)]
    #return [x[0] **2]
    #return [x[0], 0] if x[0] < 0.5 else [0.5, x[0]-0.5]
    
# Global parameters that corresponds to the structure of the network.
# Choose them according to the function you want to approximate.
nbInputs = 1
nbOutputs = 1

# The following three parameters determine which learning rates will be explored in the approximation.
# It explores startEtha first and then adds stepEtha each time, up to numberEthas times.
startEtha = 0.01
stepEtha = 0.02
numberEthas = 10

# This is the learning rate for the animation algorithm : this one remains constant
ethaAnimation = 0.05

# The number of frames of the animation.
stopAnimationAt = 100

# Here you can choose to just approximate the test function or to see the animation of the learning process.
# The animation only works with one-dimensional functions.
approximate = False
draw = True

# Imported modules

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy as cp
import time
import matplotlib.animation as animation

# Useful auxilary functions

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

class RBFnetwork:
    """An RBF neural network is caracterised by :
    - A list 'lengths' of 3 integers : 
        the number of inputs, the size of the hidden layer and the number of outputs = [n1,n2,n3]
    - A list of n2 barycentres. A barycentre is a list of n1 coefficients, each corresponding to one of the inputs
    - A list of n2 smoothing factors, each corresponding to a hidden layer's neuron.
    - A matrix of weights between the hidden layer and the output layer.
    - A column vector containing the biaises of the output layer.
    - A list of 3 layers, containing the current values inside of each neuron of the network
    - A learning rate etha
    """
    def __init__(self, lengths, etha):
        self.lengths = lengths
        initialBar = []
        initialSmooth = []
        for i in range(self.lengths[1]):
            initialBar.append(np.random.random(size = (self.lengths[0], 1)))
            #initialSmooth.append(rd.random())
            initialSmooth.append(0.5)
        self.barycentre = initialBar
        #print(self.barycentre)
        self.smoothingFactors = initialSmooth
        W = np.random.random(size=(self.lengths[2], self.lengths[1]))
        B = np.random.random(size=(self.lengths[2], 1))
        self.weights = W
        self.biaises = B
        couches = []
        for i in range(len(self.lengths)):
            couches.append(np.zeros((self.lengths[i],1)))
        self.couches = couches
        self.etha = etha

    # Computes the output that corresponds to a given input into the neural network
    def propagate(self, input):
        self.couches[0] = input
        for k in range(self.lengths[1]):
            squaredNorm = computeSquaredNorm(self.couches[0] - self.barycentre[k])
            self.couches[1][k][0] = self.phi(squaredNorm, k)
        self.couches[2] = np.dot(self.weights, self.couches[1]) + self.biaises
        return self.couches[2]

    # A function called phi, used in the calculations
    def phi(self, x, k):
        return np.exp(-self.smoothingFactors[k]*x)
    
    # The derivative of phi
    def phiprim(self, x, k):
        return -self.smoothingFactors[k]*np.exp(-self.smoothingFactors[k]*x)

    def iterationOn(self, input, expectedOutput):
        # Gets an input which it calculates the output and compares it to the expected output.
        # Then readjusts the weights, biaises, barycentres and smoothing factors accordingly.
        # This is an online-training methods as the computations are done input by input.
        output = self.propagate(input)
        expectedOutput = separate(expectedOutput)
        error = expectedOutput - output
        self.backPropagate(error)

    # Backpropagation algorithm -> modifies the parameters of the RBF network according to the error
    def backPropagate(self, error):
        deltaC = error
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] += self.etha * deltaC[i] * self.couches[1][j]
        self.biaises += self.etha * deltaC
        matrixDeriv = np.zeros((self.lengths[1], self.lengths[0]))
        for l in range(self.lengths[1]):
            for m in range(self.lengths[0]):
                deriv = 0
                for i in range(self.lengths[2]):
                    deriv+=error[i]*self.weights[i][l]
                deriv = deriv*(self.couches[0][m]-self.barycentre[l][m])
                deriv = 2*deriv*self.phiprim(computeSquaredNorm(self.couches[0]-self.barycentre[l]), l)
                matrixDeriv[l][m] = deriv
        for l in range(self.lengths[1]):
            delta = 0
            for i in range(self.lengths[2]):
                delta += error[i]*self.weights[i][l]

            self.smoothingFactors[l] += self.etha * delta
            delta *= self.phiprim(computeSquaredNorm(self.couches[0]-self.barycentre[l]), l)
        self.updateBarycentres(matrixDeriv)

    def updateBarycentres(self, matrixDeriv):
        for l in range(self.lengths[1]):
            for m in range(self.lengths[0]):
                self.barycentre[l][m] -= self.etha * matrixDeriv[l][m]
             
    '''
    First training function. The arguments are :
    - a function to approximate
    - a training set which will be used for training the networks with backpropagations
    - a validation set which is not used for training but for calculating the loss
    - a number of iterations of the whole process
    - an optional parameter that can stop the training if the loss increases between two iterations
    '''
    def train(self, function, trainingSet, validationSet, nbIterations, stopIfGrowing = False) :
        best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
        best_loss = self.getLoss(validationSet, function)
        for i in range(nbIterations) :
            #trainingSet = mix(trainingSet)
            for x in trainingSet :
                input = np.transpose(np.array([x]))
                expectedOutput = np.array([function(x)])
                self.iterationOn(input, expectedOutput)
            loss = self.getLoss(validationSet, function)
            #print("Loss is " + str(loss))
            if loss < best_loss :
                best_loss = loss
                best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
            elif stopIfGrowing :
                self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
                #print("Final loss is " + str(best_loss))
                #print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
                return
        self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
        #print(self.barycentre)
        
    '''
    A more advanced training function.
    This time the number of iterations is not fixed, but the process will stop if some conditions
    linked with time1 and time2 are verified. The training will be prolonged if the loss decreases
    fast enough, but it will necessarily stop at one point.
    The parameter divideEthaBy is the number by which the learning rate will be divided at each iteration.
    '''
    def train2(self, function, trainingSet, validationSet, batch_size, time1=20, time2=30, divideEthaBy=1) :
        old_loss = self.getLoss(validationSet, function)
        new_loss = old_loss*0.95
        best_loss = old_loss
        print("Loss is " +  str(self.getLoss(validationSet, function)))
        best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
        t0 = time.time()
        t1 = t0
        while time.time() - t0 < time1 and new_loss <= old_loss*(3 - min(2.001, (time.time() - t1)*6/time2)) :
            self.etha /= divideEthaBy
            for k in range(batch_size) :
                #trainingSet = mix(trainingSet)
                for x in trainingSet :
                    input = np.transpose(np.array([x]))
                    expectedOutput = np.array([function(x)])
                    error = expectedOutput - self.propagate(input)
                    if not selective_training or sum(sum([x**2 for x in error])) >= new_loss / (10*len(validationSet)) :
                        self.iterationOn(input, expectedOutput)
            old_loss = new_loss
            new_loss = self.getLoss(validationSet, function)
            print("Loss is " +  str(self.getLoss(validationSet, function)))
            if new_loss < best_loss :
                t1 = time.time()
                best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
                if new_loss <= 0.98*best_loss :
                    t0 = time.time()
                best_loss = new_loss
        self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
        return self
    
    '''
    Similar to train but this time there is no given function among the arguments,
    and the training set and validation set are not only lists of inputs, instead they are lists
    that each contain an input and the corresponding expected output
    '''
    def train3(self, trainingSet, validationSet, nbIterations, stopIfGrowing = False) :
        best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
        best_loss = self.getLoss3(validationSet)
        for i in range(nbIterations) :
            #trainingSet = mix(trainingSet)
            for x in trainingSet :
                input = fst(x)
                expectedOutput = snd(x)
                self.iterationOn(input, expectedOutput)
            loss = self.getLoss3(validationSet)
            print("Loss is " + str(loss))
            if loss < best_loss :
                best_loss = loss
                best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
            elif stopIfGrowing :
                self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
                print("Final loss is " + str(best_loss))
                print("Verif : final loss is " + str(self.getLoss(validationSet, function)))
                return
        self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss3(validationSet)))
    
    '''
    Similar to train but this time there is no given function among the arguments,
    and the training set and validation set are not only lists of inputs, instead they are lists
    that each contain an input and the corresponding expected output
    '''
    def train4(self, trainingSet, validationSet, batch_size, time1=20, time2=30, ethaConstant=True, divideEthaBy=1) :
        self.etha /= divideEthaBy
        old_loss = self.getLoss3(validationSet)
        new_loss = old_loss*0.95
        best_loss = old_loss
        print("Loss is " +  str(self.getLoss3(validationSet)))
        best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
        t0 = time.time()
        t1 = t0
        while time.time() - t0 < time1 and new_loss <= old_loss*(3 - min(2.001, (time.time() - t1)*6/time2)) :
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
                if new_loss <= best_loss and new_loss/old_loss >= .9 :
                    self.etha *= np.sqrt(2.02 - new_loss/best_loss)
                elif min(abs(new_loss/best_loss), abs(best_loss/new_loss)) <= .95 :
                    self.etha *= min(abs(new_loss/best_loss), abs(best_loss/new_loss))
                print("current etha is " + str(self.etha))
            if new_loss < best_loss :
                t1 = time.time()
                best_parameters = copy(self.weights), copy(self.biaises), copy(self.barycentre), copy(self.smoothingFactors)
                if new_loss <= 0.98*best_loss :
                    t0 = time.time()
                best_loss = new_loss
        self.weights, self.biaises, self.barycentre, self.smoothingFactors = best_parameters
        print("Final loss is " + str(best_loss))
        print("Verif : final loss is " + str(self.getLoss3(validationSet)))
        return self

    # Gets the loss of the vector, which is the sum of squared errors when approximated a function on a validation set
    def getLoss(self, validationSet, function) :
        
        X = [self.propagate(np.transpose(np.array([x]))) for x in validationSet]
        Y = [np.array([[function(x)]]) for x in validationSet]
        loss = sum(sum([(X[k] - Y[k])**2 for k in range(len(X))]))
        loss = sum(loss)
        loss = sum(loss)
        return loss

    # Gets the loss of the network out of a validation set that contains a list of couples
    # which first element is an output and the second element is the expected corresponding output
    def getLoss3(self, validationSet) :
        X = [self.propagate(fst(x)) for x in validationSet]
        Y = [snd(x) for x in validationSet]
        loss = sum(sum([(X[k] - Y[k])**2 for k in range(len(X))]))
        return loss



# Returns the sum of the squared elements of the vector
def computeSquaredNorm(vector):
    norm = 0
    for i in range(len(vector)):
        norm += vector[i]**2
    return norm[0]

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

# Plots the test function and its approximation by a neural network. Only relevant for one-input-one-output functions.
def plotResult(neuNet, function, trainingSet, validationSet, testSet) :
    X = [k/1000 for k in range(1000)]
    Y = [neuNet.propagate(x)[0][0] for x in X]
    Z = [function(np.transpose(np.array([x]))) for x in X]
    plt.plot(X,Y)
    plt.plot(X,Z)
    plt.show()
    
trainingSet, validationSet, testSet = createSets(nbPoints, nbInputs)

# Initiates the graphical animation.
def initiate_graphics() :
    global figure
    figure = plt.figure()
    X = [k/1000 for k in range(1000)]
    plt.plot(X, [f(np.transpose(np.array([x]))) for x in X], "blue")

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
        expectedOutput = np.array([f(x)])
        error = expectedOutput - neuNet.propagate(input)
        if not selective_training or sum(sum([x**2 for x in error])) >= new_loss / (6*len(validationSet)) :
            neuNet.iterationOn(input, expectedOutput)
    plt.plot(X, Y, color=(1, 1, 1))
    if not leaveTrace :
        for k in range(-10,10) :
            plt.plot(X, [y + k*.0002 for y in Y], color=(1,1,1))
        plt.plot(X, [f(np.transpose(np.array([x]))) for x in X], "blue")
    X = [k/200 for k in range(200)]
    Y = [neuNet.propagate(x)[0][0] for x in X]
    plt.plot(X,Y, "red")
    new_loss = neuNet.getLoss(validationSet, f)
    print("Loss is " +  str(new_loss))

# Creates an animation showing how the neural network approximates the test function over the training.
# Note that the training process is very basic here as it is done with a constant learning rate : 
# you could want to animate a more complex training (like the one of the previous procedure)
def drawAnimation(etha) :
    global figure
    global neuNet
    initiate_graphics()
    neuNet = RBFnetwork([nbInputs, 10, nbOutputs], etha)
    neuNet.train(f, trainingSet, validationSet, 1)
    anim = animation.FuncAnimation(figure, update_graphics, interval=100)
    plt.show()
    
'''
A procedure that trains a neural network to approximate the global-parametered function 'f'.
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
            neuNets.append(RBFnetwork([nbInputs, 10, nbOutputs], etha))
            neuNets[i].train(f, trainingSet, validationSet, 5, True)
            loss = neuNets[i].getLoss(validationSet, f)
            if loss < lossCand :
                lossCand = loss
                neuNetCand = neuNets[i]
                ethacand = etha
                print("New best loss is " + str(lossCand))
    print("Further training for the cand which etha is " + str(ethacand) + " and loss is " + str(lossCand))
    print("Verif : loss from neuNetCand is " + str(neuNetCand.getLoss(testSet, f)))
    neuNetCand.train2(f, trainingSet, validationSet, 1, 30, 30, 1.01)
    neuNetCand.train2(f, trainingSet, validationSet, 1, 60, 60, 1.02)
    print("Final loss with the test set is " + str(neuNetCand.getLoss(testSet, f)))
    print(neuNetCand.getLoss(testSet, f))
    if nbInputs == 1 and nbOutputs == 1 :
        plotResult(neuNetCand, f, trainingSet, validationSet, testSet)
        
if __name__ == "__main__":
    if approximate :
        procedure(startEtha, stepEtha, numberEthas)
    elif draw :
        drawAnimation(ethaAnimation)
