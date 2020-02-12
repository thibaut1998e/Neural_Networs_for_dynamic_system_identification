# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:10:35 2019

@author: Thibaut
"""

import RBF_neural_networks
import NonLinearSystemIdentificationGitHub
import scipy.io as sc
import numpy as np
import time
import random as rd
import math
import matplotlib.pyplot as plt
import copy as cp


"""On applique ici nos 2 réseaux de neurones à l'identification d'un système dynamique, le but est d'estimer la
relation qui lie les sorties et entrées aux instants antérieurs à la sortie à l'instant k. """

def getData(fileName):
    """data est une liste de couple (temps, valeur) on retourne la liste des temps et la liste des valeurs,
    le fichier d'eentrée est un fichier de type .mat
    """
    mat = sc.loadmat(fileName)
    data = list(mat.values())[3]
    return data[:,0], data[:,1]

listTemps, listForce = getData("force.mat")
#

listPosition = getData("position.mat")[1]

plt.plot(listTemps, listForce)

#plt.show()

def transform(entreeSysteme, sortieSysteme, nu, ny, nbPoints=0):
    """transforme l'entrée et la sortie du système en l'entrée et la sortie du réseau de neurones.
    Le réseau de neurones a ny + nu entrées et 1 sortie. La sortie à l'instant t dépendant des ny sorties 
    précédentes et des nu entrées précédentes
    Entrées réseaux et sorties réseaux sont une liste de vecteurs de dimensions nu + ny et 1"""
    if nbPoints != 0 :
        pas = len(entreeSysteme) // nbPoints
        i0 = (len(entreeSysteme) - pas * nbPoints) // 2
        entreeSysteme = [entreeSysteme[i0 + k*pas] for k in range(nbPoints)]
        sortieSysteme = [sortieSysteme[i0 + k*pas] for k in range(nbPoints)]
    k0 = max(ny, nu)
    entreesReseau = []
    sortiesReseau = []
    for k in range(k0, len(entreeSysteme)):

        entree = []
        for j in range(k-nu, k):
            entree.append(entreeSysteme[j])
        for j in range(k-ny, k):
            entree.append(sortieSysteme[j])
        entree = np.transpose(np.array([entree]))
        entreesReseau.append(entree)
        sortie = np.array([[sortieSysteme[k]]])
        sortiesReseau.append(sortie)

    return entreesReseau, sortiesReseau

def createSets(entreeSysteme, sortieSysteme, nu, ny, nbPoints=0) :
    """Creer les 3 sous ensembles de données, trainningSet, validationSet et TestSet, 70% pour l'entrainement,
    15% pour le test et 15% pour la validation"""
    trainingSet = []
    validationSet = []
    X, Y = transform(entreeSysteme, sortieSysteme, nu, ny, nbPoints)
    for k in range(int(len(X)*0.70)) :
        r = rd.randint(0,len(X)-1)
        trainingSet.append((X.pop(r), Y.pop(r)))
    for k in range(int(len(X)/2)) :
        r = rd.randint(0, len(X)-1)
        validationSet.append((X.pop(r), Y.pop(r)))
    testSet = [(X[i], Y[i]) for i in range(len(X))]
    return trainingSet, validationSet, testSet

def plotResult(neuNet, times, entreeSysteme, sortieSysteme, nu, ny, nbPoints=0) :
    #Trace la courbe estimée par le réseau de neurones.
    if nbPoints != 0 :
        pas = len(entreeSysteme) // nbPoints
        i0 = (len(entreeSysteme) - pas * nbPoints) // 2
        times = [times[i0 + k*pas] for k in range(nbPoints)]
    times = times[max(nu, ny):]
    X, Y = transform(entreeSysteme, sortieSysteme, nu, ny, nbPoints)
    Y = [y[0] for y in Y]  #Y est le tableau des sorties réelles et Z des sorties estimées
    Z = [neuNet.propagate(x)[0][0] for x in X]
    plt.plot(times, Y)
    plt.plot(times, Z)
    plt.show()

def MLP(nu, ny, nbNeuronesCouchesCachees, listEtha):

    nbDims = nu + ny
    batchSize = 1
    nbPoints = 0
    trainingSet, validationSet, testSet = createSets(listForce, listPosition, nu, ny, nbPoints)
    print(len(trainingSet), len(validationSet), len(testSet))
    print("Sets created")
    neuNets = []
    lossCand = 1000
    taillesCouches = [nbDims] + nbNeuronesCouchesCachees + [1]
    for i in range(len(listEtha)):
        etha = listEtha[i]
        neuNets.append(NonLinearSystemIdentificationGitHub.neuralNetWork(taillesCouches, etha))
        print("training with etha = " + str(etha))
        neuNets[i].train3(trainingSet, validationSet, batchSize, 5, True)
        loss = neuNets[i].getLoss3(validationSet)
        if loss < lossCand :
            print("New best loss is " + str(loss))
            lossCand = loss
            icand = i
            ethaCand = etha
    print("further training with etha = " + str(ethaCand))
    neuNetCand = neuNets[icand]
    listLoss = neuNetCand.train4(trainingSet, validationSet, 1, 30, 30, True, 1.04)
    listLoss2 = neuNetCand.train4(trainingSet, validationSet, 1, 60, 60, True, 1.1)


    print("Final loss with the test set is " + str(neuNetCand.getLoss3(testSet)))
    plotResult(neuNetCand, listTemps, listForce, listPosition, nu, ny, nbPoints)
    return listLoss + listLoss2

def RBF(nu, ny, nbNeuronesCoucheCachee, listEtha):

    nbDims = nu + ny
    batchSize = 1
    nbPoints = 0
    trainingSet, validationSet, testSet = createSets(listForce, listPosition, nu, ny, nbPoints)
    print(len(trainingSet), len(validationSet), len(testSet))
    print("Sets created")
    neuNets = []
    lossCand = 1000
    for i in range(len(listEtha)):
        etha = listEtha[i]
        neuNets.append(RBF_neural_networks.RBFNetWork([nbDims, nbNeuronesCoucheCachee, 1], etha))
        print("training with etha = " + str(etha))
        neuNets[i].train3(trainingSet, validationSet, batchSize, True)
        loss = neuNets[i].getLoss3(validationSet)
        if loss < lossCand:
            print("New best loss is " + str(loss))
            lossCand = loss
            icand = i
            ethaCand = etha
    print("further training with etha = " + str(ethaCand))
    neuNetCand = neuNets[icand]
    listLoss = neuNetCand.train4(trainingSet, validationSet, 1, 30, 30, True, 1.04)
    listLoss2 = neuNetCand.train4(trainingSet, validationSet, 1, 60, 60, True, 1.1)
    plotResult(neuNetCand, listTemps, listForce, listPosition, nu, ny, nbPoints)

    return neuNetCand, listLoss + listLoss2




if __name__ == "__main__":
    nu = 10  #nombre d'entrées antérieures dont dépent la sortie
    ny = 10  #nombre de sorties antérieures dont dépend la sortie
    nbNeuronesCouchesCachees = [15] #pour le MLP on peut mettre plusieurs couches cachees, si vous voulez mettre n
    #couches cachees, n éléments dans cette liste  l'élément i correspond au nombre de neurones de la ieme couche cachee
    nbNeuronesCoucheCachee = 15 # pour le RBF une seule couche cachee
    listEtha = np.arange(0.001, 0.022, 0.002) #la convergence et vitesse de convergence dépend bcp de etha, on met dans cette liste
    #l'ensemble des ethas qu'on veut tester sur qqs itérations avant d'entrainer plus longuement le réseau avec le meilleur.
    neuNet, listLossMLP = MLP(nu, ny, nbNeuronesCouchesCachees, listEtha) #neuNet : le réseau après entrainement
    #neuNet, listLossRBF = RBF(nu,ny,nbNeuronesCoucheCachee,listEtha)

    plt.plot(range(len(listLossMLP)), listLossMLP) #trace la loss







