import numpy as np 
import scipy as sp
import math
import matplotlib.pyplot as plt
from hmmlearn import hmm

"""
Attempt to create a 2-D continuous particle filter
with discrete time steps

V = Valence
A = Arousal

each cross section in time would show the variation of annotations as a beta distribution

CDF takes in a value and gives ou

Initial Pdf: A normal distribution bounded to -1 and 1 (stationary)

Transmission Pdf: Various normal distributions parameterised by the previous state/sample

Emission Probability: {Function} probability distribtution parameterised by current state
    takes in observed outpt as input

Importance Pdf: {Function} probability distribution parameterised by previous state and observed output
    takes in current state as input
 
"""

class particleFilter:
    """
    Particle Filter class to store probability distributions
    """
    def __init__(self, initPdf, transPdf, emissionPdf, importancePdf):
        self._init_pdf = initPdf
        self._trans_pdf = transPdf
        self._emission_prob = emissionPdf
        self._importance_prob = importancePdf
        
        
def createNormPdf(maxVal, minVal, meanVal, stdDev):    
    initShapeA, initShapeB = (minVal - meanVal) / stdDev, (maxVal - meanVal) / stdDev
    initPdf = sp.stats.truncnorm(initShapeA, initShapeB, loc=meanVal, scale=stdDev)
    return initPdf


def getCorrespondingMean(state):
    '''
    some random piecewise function to give varying means
    
    should go low -> medium -> high {loop}
    '''
    if state < -0.2:
        return 0
    elif state < 0.2:
        return 0.4
    else:
        return -0.4

    
def getEmissionProb(currState, observedOutput):
    '''
    Set this as a random piece wise function for simplicity
    
    uniform probability of seeing any output given the observed output
    as random.rand() returns a number : [0, 1)
    
    doesn't really make sense because the integral of these probabilities over the values >> 1.....
    '''
    if currState < -0.15:
        if observedOutput < 0.33:
            return 0.7
        elif observedOutput < 0.66:
            return 0.15
        else:
            return 0.15
    elif currState < 0.15:
        if observedOutput < 0.33:
            return 0.15
        elif observedOutput < 0.66:
            return 0.7
        else:
            return 0.15
    else:
        if observedOutput < 0.33:
            return 0.15
        elif observedOutput < 0.66:
            return 0.15
        else:
            return 0.7


def getImportanceProb(prevState, observedOutput, currState):
    '''
    Set this as a random piece wise function for simplicity
    
    uniform probability of seeing any output given the observed output
    as random.rand() returns a number : [0, 1)
    
    doesn't really make sense because the integral of these probabilities over the values >> 1.....
    basically replicated a matrix here 
    '''
    if prevState < -0.15:
        if observedOutput < 0.33:
            if currState < -0.15:
                return 0.7
            elif currState < 0.15:
                return 0.15
            else:
                return 0.15
        elif observedOutput < 0.66:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.7
            else:
                return 0.15
        else:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.15
            else:
                return 0.7
    elif prevState < 0.15:
        if observedOutput < 0.33:
            if currState < -0.15:
                return 0.7
            elif currState < 0.15:
                return 0.15
            else:
                return 0.15
        elif observedOutput < 0.66:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.7
            else:
                return 0.15
        else:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.15
            else:
                return 0.7
    else:
        if observedOutput < 0.33:
            if currState < -0.15:
                return 0.7
            elif currState < 0.15:
                return 0.15
            else:
                return 0.15
        elif observedOutput < 0.66:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.7
            else:
                return 0.15
        else:
            if currState < -0.15:
                return 0.15
            elif currState < 0.15:
                return 0.15
            else:
                return 0.7
    

def propose(anEvidence:np.array, aNParticles=100):
    
    maxV = 1
    minV = -1
    stdDev = 0.7
    
    
    estimation_bound = stdDev/3
    
    initMeanV = 0
    initDistr = createNormPdf(maxV, minV, initMeanV, stdDev)
    
    transDistr = None
    emissMatr = None
    importanceDistr = None
    
    myWeights = np.zeros(aNParticles)
    myParticles = np.zeros((len(anEvidence), aNParticles), dtype=float)
    particleChoices = [x for x in range(aNParticles)]

    base_samples = []
    for i in range(aNParticles):
        myParticles[0][i] = initDistr.rvs()
    
    base_samples.append(myParticles[0][0])
    
    for tstep in range(1, len(anEvidence)):
        # propagate particles forward in one time step
        for i in range(aNParticles):
            
            # Use the previous state as a parameter for the state transition probabilities
            # as outlined above, should be consistent, sample from this new transition distribution
            # the distribution is the probability for the given previous state value
            transDistrMean = getCorrespondingMean(myParticles[tstep-1][i])
            transDistr = createNormPdf(maxV, minV, transDistrMean, stdDev)
            myParticles[tstep][i] = transDistr.rvs()
            
            # Use a rough estimate for probabilities as continuous random variables
            # don't have a spot probability
            # Super hacky code need to ask Vidhya
            g = getEmissionProb(myParticles[tstep][i], anEvidence[tstep])
            f = transDistr.cdf(myParticles[tstep][i]+estimation_bound) - transDistr.cdf(myParticles[tstep][i] - estimation_bound)
            q = getImportanceProb(myParticles[tstep-1][i], anEvidence[tstep], myParticles[tstep][i])
            
            myWeights[i] = g * f / q
        
        base_samples.append(myParticles[tstep][0])
        myWeights = myWeights / np.sum(myWeights)
        resampledParticles = np.random.choice(particleChoices, p=myWeights, size=aNParticles)
        
        for i in range(aNParticles):
            myParticles[:, i] = myParticles[:, resampledParticles[i]]           
    
    
    print(myParticles)
    
    filtered_samples = []
    for tstep in range(len(anEvidence)):
        filtered_samples.append(myParticles[tstep][0])
    
    plt.plot(filtered_samples)
    plt.plot(base_samples)
    plt.ylim([minV,maxV])
    plt.show()
    


def testFunc():
    '''
    maxV = 1
    minV = -1
    initMeanV = 0
    initStdDevV = 0.3
    
    initPdf = createNormPdf(maxV, minV, initMeanV, initStdDevV)
    x_range = np.linspace(-2,2,1000)
    # plt.plot(x_range, initPdf.pdf(x_range))
    
    samples = []
    for _ in range(10):
        samples.append(initPdf.rvs())

    print(initPdf.pdf(0.3))
    
    #plt.plot(x_range, initPdf.pdf(x_range))
    #plt.hist(samples, density=True, bins=100)
    #plt.show()
    '''
    
    
    myEvidence = np.random.rand(100)
    propose(anEvidence=myEvidence, aNParticles=100)
    
    
    

testFunc()