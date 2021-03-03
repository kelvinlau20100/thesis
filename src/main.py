import numpy as np 
from hmmlearn import hmm

def main():
    startProb = np.array([0.6, 0.3, 0.1])

    transMatr = np.array([[0.7, 0.2, 0.1],
                          [0.3, 0.5, 0.2],
                          [0.3, 0.3, 0.4]])

    emissionMatr = np.array([[0.1, 0.1, 0.1, 0.3, 0.4],
                             [0.4, 0.4, 0.1, 0.1, 0.0],
                             [0.25, 0.2, 0.2, 0.15, 0.1]])


    """
    HMM testing code
    """
    model = hmm.MultinomialHMM(n_components=3)
    model.startprob_ = startProb
    model.transmat_ = transMatr
    model.n_features = 5
    model.emissionprob_ = emissionMatr
    predictedFeatures, predictedStateSeq = model.sample(100)
    print(predictedFeatures)
    print(predictedStateSeq)
    
    
if __name__=="__main__":
    main()
