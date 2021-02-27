import numpy as np 
import scipy as sp
import math
from hmmlearn import hmm

class particleFilter:
    """
    Particle Filter class to store Starting Distribution, Transition Matrix & Emission Matrix
    """
    def __init__(self, startProb, transMatr, emissionMatr):
        self.startprob_ = startProb
        self.transmat_ = transMatr
        self.emissionprob_ = emissionMatr

def logSumExp(anArray : np.array):
    """
    Calculate the LSE of an array

    :param anArray: NumPy array of values
    :returns: LSE val
    """
    return sp.special.logsumexp(anArray)

def logMeanExp(anArray : np.array):
    """
    Calculate the LME of an array

    :param anArray: NumPy array of values
    :returns: LME val
    """
    m = anArray.max()
    V = np.exp(anArray - m)
    return m + np.log(np.mean(V))

def propose(aPf : particleFilter, anEvidence : np.array, aNParticles=100, aThreshold=75):
    """
    Run a particle filter given the set of parameters to get a trajectory of states
    
    :param aPf: particleFilter w/ Starting Distro, Transition Matrix & Emission Matrix
    :param anEvidence: NumPy array with observed outputs
    :param aNParticles: Number of particles to use in the filter
    :param aThreshold: Threshold for reweighting particles
    :returns: the log likelihood, the final particles & estimate of state trajectory
    """
    myStartProb = aPf.startprob_
    myTransMatr = aPf.transmat_
    myEmissMatr = aPf.emissionprob_
    myLogLikelihood = 0.0
    myLogWeights = np.zeros(aNParticles)
    myLogWeightsNorm = np.zeros(aNParticles)
    myParticles = np.zeros((len(anEvidence), aNParticles), dtype=int)


    myParticles[0,:] = np.random.choice([0, 1, 2], size=aNParticles, p=myStartProb)

    print(myParticles.shape)
    for t in range(1, len(anEvidence)):

        # Propagate my particles forward one time step
        for i in range(0, aNParticles):
            myParticles[t, i] = np.random.choice([0, 1, 2], p=myTransMatr[myParticles[t-1, i]])
            print("t-1 state :", myParticles[t-1, i], ", t state :", myParticles[t, i])


###################################################################################################
def testFuncs():
    #print("log sum exp : ", logSumExp(np.array([5, 4, 3, 2, 1, 0])))
    #print("log mean exp : ", logMeanExp(np.array([0, 1, 2, 3, 4, 5])))
    startProb = np.array([0.3, 0.3, 0.4])

    transMatr = np.array([[0.1, 0.45, 0.45],
                        [0.45, 0.1, 0.45],
                        [0.45, 0.45, 0.1]])

    emissionMatr = np.array([[0.1, 0.1, 0.1, 0.3, 0.4],
                            [0.4, 0.4, 0.1, 0.1, 0.0],
                            [0.25, 0.2, 0.2, 0.15, 0.1]])

    myPf = particleFilter(startProb, transMatr, emissionMatr)
    myEvidence = np.random.randint(low=0, high=5, size=200)

    propose(aPf=myPf, anEvidence=myEvidence)


testFuncs()