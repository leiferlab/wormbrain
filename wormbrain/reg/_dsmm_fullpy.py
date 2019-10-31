import numpy as np
from scipy.special import gamma as spgamma
from scipy.special import digamma as spdigamma
from scipy.special import polygamma as sppolygamma
from scipy.special import bernoulli as spbernoulli
from scipy.optimize import root as sproot
from scipy.optimize import root_scalar as sproot_scalar
from wormbrain.match import pairwise_distance
import mistofrutta as mf
#import multiprocessing as mp
#import time
#import matplotlib.pyplot as plt

def _neighborhood(pwise_dist,cutoff=2.): 
    D = pwise_dist
    mediana = np.average(np.sort(D,axis=0)[1,:])
    neighborWeights = np.where(D<cutoff*mediana,1.,0.)
    neighborN = np.sum(neighborWeights,axis=0)
    
    return neighborN, neighborWeights
            
def _studt(sigma2,Gamma,D,pwise_dist):
    A = spgamma(0.5*(Gamma+D))
    B = np.sqrt(2.*sigma2) * (Gamma+spgamma(0.5))**(0.5*D) * spgamma(0.5*Gamma)
    C = (1.0 + pwise_dist*(1./(sigma2*Gamma[:,None])))**(0.5*(D+Gamma[:,None]))
    
    return A[:,None]/(B[:,None]*C)
    
def _eqforgamma(Gamma,CDE_term):
    Gammahalves = 0.5*Gamma
    AB = -spdigamma(Gammahalves) + np.log(Gammahalves)
    
    out = np.empty_like(Gammahalves)
    mf.approx.trigamma(Gammahalves,len(Gammahalves),out,9)
    jac = -0.5*out + 1./Gamma
    Jac = np.diag(jac)

    return (1.+AB+CDE_term), Jac
    
def _eqforalpha(alpha,p,sumPoverN):
    #Eq. (20)', switching indices for x and y to match their first paper. 
    #Added an alpha to the first term inside the \biggl ( in the final line!!
    #Is it reasonable to use an approximate form for the exponential?
    #print(alpha*np.max(sumPoverN)) #around 0.2 max
    #print(alpha*np.min(sumPoverN)) # 0 (always positive, right?)
    A_term = np.empty_like(sumPoverN)
    #mf.approx.exp(alpha*sumPoverN,np.prod(sumPoverN.shape),A_term,0)
    A_term = np.exp(alpha*sumPoverN) # TODO approximate this exponential
    B_term = np.sum(alpha* sumPoverN * A_term,axis=1) #There's an alpha here or not???
    C_term = np.sum(A_term,axis=1)
    D_term = -B_term/C_term
    
    return np.sum(p*(1.*sumPoverN+D_term[:,None]))#alpha

var_dict = {}

def init_worker(A_rawarr, B_rawarr, kwargs):
    for key in kwargs:
        var_dict[key] = kwargs[key]
    var_dict['A_rawarr'] = A_rawarr
    var_dict['B_rawarr'] = B_rawarr
    
def dsmm(A,B,**kwargs):
    # If multiple A pointsets are passed, run the registrations in parallel
    if len(A.shape)==3:
        print("parallel")
        if __name__ == '__main__':
            cores = kwargs.pop('cores')
            A_rawarr = mp.RawArray('d', int(np.prod(A.shape)))
            A_rawarr_np = np.frombuffer(A_rawarr, dtype=np.float64).reshape(A.shape)
            np.copyto(A_rawarr_np, A)
            B_rawarr = mp.RawArray('d', int(np.prod(B.shape)))
            B_rawarr_np = np.frombuffer(B_rawarr, dtype=np.float64).reshape(B.shape)
            np.copyto(B_rawarr_np, B)
            
            kwargs['Ashape'] = A.shape
            kwargs['Bshape'] = B.shape
            
            pool = mp.Pool(processes=cores, initializer=init_worker, 
                                initargs=(A_rawarr, B_rawarr, kwargs))
            
            P = pool.map(_dsmm_parallel_wrapper, range(A.shape[0]))
            pool.close()
            pool.join()
            return P
        
    elif len(A.shape)==2:
        return _dsmm(A,B,**kwargs)
        
def _dsmm_parallel_wrapper(i):
    #a = np.prod(var_dict['Ashape'][1:])
    #A = np.frombuffer(var_dict['A_rawarr'])[a*i:a*(i+1)].reshape(var_dict['Ashape'])
    #B = np.frombuffer(var_dict['B_rawarr']).reshape(var_dict['Bshape'])
    
    A = i[0]
    BB = i[1]
    for B in BB:
        _dsmm(A, B,returnOnlyP=True)
    #print(i)
    
    print("Hi")
    return _dsmm(A, B,returnOnlyP=True)

def _dsmm_fullpy(Y,X,beta=2.0,llambda=1.5,neighbor_cutoff=10.0,gamma0=3.0,
            conv_epsilon=1e-3,eq_tol=1e-2,returnAll=False):
    '''(Version implemented fully in Python. For the most efficient one, see 
    dsmmc in _dsmm_c_py.py and _dsmm_c.cpp)
    Registers Y onto X via a nonrigid pointset registration based on a 
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
        Default: 1e-2
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
    # This is the version running with a single pair of pointsets.
    # This registers Y onto X (Y is what is changed)
    # In wormb.match, I always pass A,B, where B is the refBrain.
    
    # Preprocess ("normalize" in Vemuri's language)
    
    X -= np.average(X,axis=0)
    maxX = np.max(np.absolute(X),axis=0)
    if np.all(maxX>0): X /= maxX
    Y -= np.average(Y,axis=0)
    maxY = np.max(np.absolute(Y),axis=0)
    if np.all(maxY>0): Y /= maxY
    
    N = X.shape[0]
    M = Y.shape[0]
    D = X.shape[1]
    
    # Init parameters
    pwise_dist = pairwise_distance(Y,X,squared=True)
    #mf.approx.pwise_dist2(Y,X,M,N,D,pwise_dist)
    beta2 = beta**2
    w = np.ones((M,N))*(1./M/N)
    Gamma = np.ones(M)*gamma0
    Gamma_old = np.copy(Gamma)
    sigma2 = np.sum(pwise_dist)/(D*M*N)
    pwise_distYY = pairwise_distance(Y,Y,squared=True)
    G = np.exp(-pwise_distYY/(2.*beta2))
    Identity = np.diag(np.ones(M))

    # Allocate arrays
    F_t = np.empty((M,N))
    wF_t = np.empty((M,N))
    wF_t_sum = np.empty(N)
    p = np.empty((M,N))
    u = np.empty((M,N))
    CDE_term = np.empty(M)
    hatP = np.empty((M,N))
    hatPI = np.empty((M,M))
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
    #neighborN, neighborWeights = _neighborhood(pwise_distYY,neighbor_cutoff)
    
    # Convergence
    relerr = 1000.
    regerror = np.sum(pairwise_distance(X,Y,squared=True))
    
    i = 0
    ##### array[y,x]
    while relerr > conv_epsilon:
        #Step3 (Eq. (5))
        F_t[:] = _studt(sigma2,Gamma,D,pwise_dist)

        #Step3:E-Step
        #Eq. (17)'
        wF_t = w*F_t
        p[:] = wF_t / np.sum(wF_t,axis=0)[None,:]
        #p = np.absolute(p)
        
        #Eq. (16) and (21)'
        u[:] = (Gamma[:,None] + D) / (Gamma[:,None] + pwise_dist/sigma2)
        #u = np.absolute(u)

        #Eq. (20)'
        neighborN, neighborWeights = _neighborhood(pwise_distYY,neighbor_cutoff)
        sumPoverN = np.sum(neighborWeights[...,None]*p[None,...],axis=1)/neighborN[:,None] ##############TODO this takes a long time! 2.5 ms! It's because I augment the dimensionality so much.
        
        Result = sproot(_eqforalpha,x0=alpha,args=(p,sumPoverN),method="hybr",tol=eq_tol)
        alpha = Result['x'][0]
        #Result = sproot_scalar(_eqforalpha,x0=alpha,args=(p,sumPoverN),method="brentq",bracket=[0.05,10.],rtol=eq_tol)
        #alpha = Result.root
        
        #Step4:M-Step
        # Eq. (18)'
        expAlphaSumPoverN = np.exp(alpha*sumPoverN)
        w[:] = expAlphaSumPoverN*(1./np.sum(expAlphaSumPoverN,axis=0)[None,:])
        
        #Eq. (23)
        Gamma_old = np.copy(Gamma)
        # C,D,E_terms were terms in the function _eqforgamma, but since they 
        # never change, I calculate them just once out here. 
        Gammaoldpdhalves = np.absolute(0.5*(Gamma_old+D))
        C_term = np.sum(p*(np.log(u)-u),axis=1)/np.sum(p,axis=1)
        D_term = spdigamma(Gammaoldpdhalves)
        E_term = -np.log(Gammaoldpdhalves)
        CDE_term = C_term + D_term + E_term
        
        Result = sproot(_eqforgamma,x0=Gamma_old,args=(CDE_term),method="hybr",tol=eq_tol,jac=True,options={'col_deriv':1})
        '''for mm in np.arange(len(Gamma)):
            for jj in np.arange(100):
                d = (2.*np.exp(spdigamma(Gamma[mm]*0.5)-1.-CDE_term[mm])-Gamma[mm])
                d *= -10.
                if(CDE_term[mm]<-2.): d /= float(jj)
                if(np.abs(d)<0.0001): break
                Gamma[mm] = max(0.1,Gamma[mm]+d)'''
        Gamma = Result['x']

        #Eq. (26)
        hatP[:] = p*u
        hatPI_diag[:] = np.sum(hatP,axis=1)
        G[:] = np.exp(-0.5/beta2*pwise_distYY) 
        hatPIG[:] = hatPI_diag[:,None]*G
        
        hatPIY = hatPI_diag[:,None]*Y
         
        hatPX[:] = np.dot(hatP,X)
        A = np.linalg.inv(hatPIG+llambda*sigma2*Identity)
        B = hatPX - hatPIY
        W = np.dot(A,B)
        #Step5 moved here to optimize
        Y += np.dot(G,W)
        pwise_dist = pairwise_distance(Y,X,squared=True) 
        pwise_distYY = pairwise_distance(Y,Y,squared=True)
        #Back to Step4 
        #Eq. (27)
        AA = np.sum(hatP*pwise_dist)#np.sum((X[None,:]-Y[:,None])**2,axis=-1))
        BB = D*np.sum(hatP) # or just p as in the old paper?
        sigma2 = AA*(1./BB)
        
        #Step6
        regerror_old = regerror
        regerror = np.sum(pwise_dist)
        relerr = np.absolute((regerror-regerror_old)/regerror_old)
        
        i += 1
    
    Conf = np.max(p,axis=1)
    Match = np.where(Conf>0.5,np.argmax(p,axis=1),-1)
    
    if returnAll:
        return Y,X,p,Match
    else:    
        return Match
