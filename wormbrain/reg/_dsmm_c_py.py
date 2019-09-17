import wormbrain as wormb
import numpy as np

def dsmmc(Y,X,returnOnlyP=False,beta=2.0,llambda=1.5,neighbor_cutoff=10.0,
            gamma0=3.,conv_epsilon=1e-3,eq_tol=1e-4):
    
    '''
    # Preprocess ("normalize" in Vemuri's language)
    X -= np.average(X,axis=0)
    X /= np.max(np.absolute(X),axis=0)
    #X += np.min(X,axis=0)
    Y -= np.average(Y,axis=0)
    Y /= np.max(np.absolute(Y),axis=0)
    #Y += np.min(Y,axis=0)
    '''

    N = X.shape[0]
    M = Y.shape[0]
    D = X.shape[1]

    # Init parameters
    #pwise_dist = wormb.match.pairwise_distance(Y,X,squared=True)
    #pwise_distYY = wormb.match.pairwise_distance(Y,Y,squared=True)
    #w = np.ones((M,N))*(1./M/N)
    #Gamma = np.ones(M)*3.
    #beta2 = beta**2
    #G = np.exp(-pwise_distYY/(2.*beta2))
    #Identity = np.diag(np.ones(M))

    # Allocate arrays
    '''F_t = np.empty((M,N))
    wF_t = np.empty((M,N))
    wF_t_sum = np.empty((M,N))
    p = np.empty((M,N))
    u = np.empty((M,N))
    CDE_term = np.empty(M)
    hatP = np.empty((M,N))
    hatPI_diag = np.empty(M)
    hatPIG = np.empty((M,M))
    hatPX = np.empty((M,D))
    hatPIY = np.empty((M,D))
    W = np.empty((M,D))
    GW = np.empty((M,D))'''
    
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
    alpha=0.2

    wormb.reg._dsmmc_bare(X,Y,M,N,D,beta,llambda,neighbor_cutoff,alpha,gamma0,
           conv_epsilon,eq_tol,
           pwise_dist,pwise_distYY,Gamma,CDE_term,
           w,F_t,wF_t,wF_t_sum,p,u,Match,
           hatP,hatPI_diag,hatPIG,hatPX,hatPIY,
           G,W,GW,sumPoverN,expAlphaSumPoverN)
              
    return Y,X,p,Match
