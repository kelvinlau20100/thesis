import numpy as np 
import scipy as sp
import math
from hmmlearn import hmm

class particleFilter:
    def __init__(self, startProb, transMatr, emissionMatr):
        self.startprob_ = startProb
        self.transmat_ = transMatr
        self.emissionprob_ = emissionMatr

def logSumExp(anArray : np.array):
    """
    Helper Function to calculate the LSE of an array
    inputs  : NumPy array of values
    outputs : LSE value
    """
    return sp.special.logsumexp(anArray)

def logMeanExp(anArray : np.array):
    """
    Helper function to calculate the LME of an array
    inputs  : NumPy array of values
    outputs : LME val
    """
    m = anArray.max()
    V = np.exp(anArray - m)
    return m + np.log(np.mean(V))

def propose(pf, evidence, nParticles=100, threshold=75):

    pass

print("log sum exp : ", logSumExp(np.array([5, 4, 3, 2, 1, 0])))

print("log mean exp : ", logMeanExp(np.array([0, 1, 2, 3, 4, 5])))

startProb = np.array([0.6, 0.3, 0.1])

transMatr = np.array([[0.7, 0.2, 0.1],
                      [0.3, 0.5, 0.2],
                      [0.3, 0.3, 0.4]])

emissionMatr = np.array([[0.1, 0.1, 0.1, 0.3, 0.4],
                         [0.4, 0.4, 0.1, 0.1, 0.0],
                         [0.25, 0.2, 0.2, 0.15, 0.1]])


myPf = particleFilter(startProb, transMatr, emissionMatr)

