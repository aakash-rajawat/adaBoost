import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    """
    Adaboost with least squares linear classifier as weak classifier on USPS data
    for a high dimensional dataset
    :param X:   the dataset (numSamples x numDim)
    :param Y:   labeling    (numSamples x 1)
    :param K:   number of weak classifiers (scalar)
    :param nSamples: number of data points obtained by weighted sampling (scalar)
    :param percent: percentage of the data set that is used as test data set (scalar)
    :return:
        alphaK: voting weights (1 x k)
        para:   parameters of simple classifier (K x (D+1))
        error:  training error (1 x k)
    """
    # Sample random a percentage of data as test data set
    N = len(X)
    numb = round( N * percent)
    pos = choice(N, numb, False)
    allpos = range(N)
    restpos = np.setdiff1d(allpos, pos)

    testX = X[pos]
    testY = Y[pos]
    newX = X[restpos]
    newY = Y[restpos]
    X = newX
    Y = newY

    # Initialization
    n = N - numb

    w = (np.ones(n)/n).reshape(n, 1)
    alphaK = np.ones(K)
    error = np.ones(K)
    para = np.ndarray((K, X.shape[1]+1))

    #initialize loop
    for k in range(K):
        
        # weight sampling of data
        #print(w.shape)
        index = choice(n,nSamples,True,w.ravel())
        X_sampled = X[index]
        Y_sampled = Y[index]

       # Train the weak classifier Ck
        weights, bias = leastSquares(X_sampled, Y_sampled)

        para[k,:] = np.append(weights, [bias])

        # classify
        cY = np.sign(np.append(np.ones(n).reshape(n,1), X, axis = 1).dot(para[k].T)).T

        # calculate error for given classifier
        temp = np.where([Y[i] != cY[i] for i in range(n)], 1, 0).reshape(n, 1)
        ek = np.sum(w * temp)

        # If the error is zero, the data set is correct classified - break the loop
        if ek < 1.0e-01:
            alphaK[k]=1
            break

        # Compute the voting weight for the weak classifier alphak
        alphaK[k] = 0.5 * np.log((1 - ek) / ek)

        # recalculate the weights
        w = w * np.exp(-alphaK[k] * (Y*(cY.reshape(len(cY),1)))) #TODO: check mult
        w = w/sum(w)

        # calculate error for boosted classifier
        classlabels, _ = eval_adaBoost_leastSquare(testX, alphaK[:k+1], para[:k+1])
        classlabels = classlabels.reshape(len(classlabels), 1)

        error[k] = sum([classlabels[i] != testY[i] for i in range(len(classlabels))])/len(testY)

    return [alphaK, para, error]
