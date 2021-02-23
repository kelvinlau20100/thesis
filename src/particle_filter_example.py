import numpy as np 
from hmmlearn import hmm

def main():
    startProb = np.array([0.6, 0.3, 0.1])

    transMatr = np.array([[0.7, 0.2, 0.1],
                         [0.3, 0.5, 0.2],
                         [0.3, 0.3, 0.4]]
                        )

    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = startProb
    model.transmat_ = transMatr
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(100)

    print(Z)
    

if __name__=="__main__":
    main()
