import numpy as np 
import scipy as sp
import math
from hmmlearn import hmm

class particleFilter:
    """
    Particle Filter class to store Starting Distribution, Transition Matrix & Emission Matrix
    """
    def __init__(self, startProb, transMatr, emissionMatr, importanceDistribution):
        self.startprob_ = startProb
        self.transmat_ = transMatr
        self.emissionprob_ = emissionMatr
        self.importancedistr_ = importanceDistribution

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
    myImportanceDistribution = aPf.importancedistr_
    myLogLikelihood = 0.0
    myWeights = np.zeros(aNParticles)
    myParticles = np.zeros((len(anEvidence), aNParticles), dtype=int)
    particleChoices = [x for x in range(aNParticles)]

    myParticles[0,:] = np.random.choice([0, 1, 2], size=aNParticles, p=myStartProb)

    for t in range(1, len(anEvidence)):
        # Propagate my particles forward one time step
        for i in range(0, aNParticles):
            # Iteration through each individual particle
            # Find the next state
            # Calculate the weights of the particle
            # need to calculate 3 things for each particle, g(y_n | X_n) , f(X_n | X_n-1), q(X_n | y_n, X_n-1)
            
            myParticles[t, i] = np.random.choice([0, 1, 2], p=myTransMatr[myParticles[t-1, i]])
            g = myEmissMatr[myParticles[t,i]][anEvidence[t]]
            f = myTransMatr[myParticles[t-1, i]][myParticles[t,i]]
            q = myImportanceDistribution[anEvidence[t]][myParticles[t-1,i]][myParticles[t,i]]
            myWeights[i] = g * f / q
            
            # Debug outputs
            # print("X_n = ", myParticles[t,i], "y_n = ", anEvidence[t])
        
        # normalise the weights to resample the particle paths
        myWeights = myWeights / np.sum(myWeights)
        resampledParticles = np.random.choice(particleChoices, p=myWeights, size=aNParticles)
        
        for i in range(0, aNParticles):
            myParticles[:, i] = myParticles[:, resampledParticles[i]]
    
    print(myParticles)
    print(myWeights)
    # Can draw a trajectory path by sampling from the final set of weights and taking that particle path from
    # myParticles
            
            
###################################################################################################
def testFuncs():
    #print("log sum exp : ", logSumExp(np.array([5, 4, 3, 2, 1, 0])))
    #print("log mean exp : ", logMeanExp(np.array([0, 1, 2, 3, 4, 5])))
    
    # Currently this particle filter is for a single observation so Evidence is a 1-D array
    startProb = np.array([0.3, 0.3, 0.4])

    transMatr = np.array([[0.1, 0.45, 0.45],
                        [0.45, 0.1, 0.45],
                        [0.45, 0.45, 0.1]])

    emissionMatr = np.array([[0.1, 0.1, 0.1, 0.3, 0.4],
                            [0.4, 0.4, 0.1, 0.1, 0.0],
                            [0.25, 0.2, 0.2, 0.15, 0.1]])
    
    importanceDistribution = np.zeros((5, 3, 3))
    for i in range(5):
        for j in range(3):
            importanceDistribution[i][j] = [0.2, 0.6, 0.2]
    

    myPf = particleFilter(startProb, transMatr, emissionMatr, importanceDistribution)
    myEvidence = np.random.randint(low=0, high=5, size=20)

    propose(aPf=myPf, anEvidence=myEvidence, aNParticles=6)

testFuncs()
