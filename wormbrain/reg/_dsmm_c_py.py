import wormbrain as wormb
import numpy as np
import matplotlib.pyplot as plt

def dsmmc(Y,X,beta=2.0,llambda=1.5,neighbor_cutoff=10.0,
            gamma0=3.,conv_epsilon=1e-3,eq_tol=1e-4,returnAll=False):
    '''Registers Y onto X via a nonrigid pointset registration based on a 
    Student's t-distribution mixture model with Dirichlet-distribution priors
    via an expectation-maximization algorithm.
    Ref: doi:10.1371/journal.pone.0091381 and doi:10.1038/s41598-018-26288-6
    
    Parameters
    ----------
    Y, X: numpy array
        Sets of points in D-dimensional space (Y gets moved onto X). These 
        arrays are modified: if you need to keep the original ones, pass copies.
        Note: Should be contiguous row-major arrays, with indices 
        [point, coordinate].
    beta: float, optional
        Standard deviation of Gaussian smoothing filter. See equations in the 
        references. Default: 2.0
    llambda: float, optional
        Regularization parameter. See equations in the references. Default: 1.5
    neighbor_cutoff: float, optional
        Multiple of the average nearest-neighbor distance within which points
        are considered neighbors. See equations in the references. Default: 10.0
    gamma0: float, optional
        Initialization of the gamma_m parameters (degrees of freedom of the
        Student's t-distribution). See equations in the references. Default: 3.0
    conv_epsilon: float, optional
        Relative error on the displacements of the points in Y at which the 
        algorithm is considered at convergence. Default: 1e-3
    eq_tol: float, optional
        Tolerance for convergence of the numerical solution of the equations
        for gamma_m and \\bar alpha. See equations in the references. 
        Default: 1e-4
    returnAll: boolean, optional
        If True, the function returns Y, X, p, Match. See below. If False, it 
        only returns Match.
        
    Returns
    -------        
    Y, X: numpy array
        Same as input.
    p: numpy array
        p[m,n] is the posterior probability for the match of Y[m] to X[n].
    Match: numpy array
        X[Match[m]] is the point in X to which Y[m] has been matched. The 
        built-in criterion is that the maximum posterior probability p[m,:] for 
        Y[m] has to be greater than 0.5. If a different criterion is needed, set
        returnAll to True and use the returned p to calculate the matches.
    '''
    N = X.shape[0]
    M = Y.shape[0]
    D = X.shape[1]

    # Allocate arrays
    
    pwise_dist = np.empty((M,N))
    pwise_distYY = np.empty((M,M))
    w = np.empty((M,N))
    Gamma = np.empty(M)
    G = np.empty((M,M))
    F_t = np.empty((M,N))
    wF_t = np.empty((M,N))
    wF_t_sum = np.empty(N)
    p = np.empty((M,N))
    u = np.empty((M,N))
    Match = np.ones(M,dtype=np.int32)*(-10)
    CDE_term = np.empty(M)
    hatP = np.empty((M,N))
    hatPI_diag = np.empty(M)
    hatPIG = np.empty((M,M))
    hatPX = np.empty((M,D))
    hatPIY = np.empty((M,D))
    W = np.empty((M,D))
    GW = np.empty((M,D))

    #Additional stuff from new paper
    sumPoverN = np.zeros((M,N)) 
    expAlphaSumPoverN = np.zeros((M,N))
    alpha=1.

    wormb.reg._dsmmc_bare(X,Y,M,N,D,beta,llambda,neighbor_cutoff,alpha,gamma0,
           conv_epsilon,eq_tol,
           pwise_dist,pwise_distYY,Gamma,CDE_term,
           w,F_t,wF_t,wF_t_sum,p,u,Match,
           hatP,hatPI_diag,hatPIG,hatPX,hatPIY,
           G,W,GW,sumPoverN,expAlphaSumPoverN)
    
    if returnAll:         
        return Y,X,p,Match
    else:
        return Match
