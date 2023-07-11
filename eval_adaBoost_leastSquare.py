import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    """

    :param X:      test data points (numSamples x numDim)
    :param alphaK: classifier voting weights (K x 1)
    :param para:   parameters of simple classifier (K x (D +1))
    :return:
        classLabels: labels for data points (numSamples x 1)
        result:      weighted sum of all the K classifier (scalar)
    """
    K = para.shape[0]
    N = X.shape[0]
    result = np.zeros(N)

    for k in range(K):
        cY = np.sign(np.append(np.ones(N).reshape(N,1), X, axis = 1).dot(para[k])).T
        result = result + cY.dot(alphaK[k])

    classLabels = np.sign(result)

    return [classLabels, result]
