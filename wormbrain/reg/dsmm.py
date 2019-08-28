import numpy as np
from scipy.special import gamma as spgamma
from scipy.special import digamma as spdigamma
from scipy.special import polygamma as sppolygamma
from scipy.optimize import root as sproot
from wormbrain.match import pairwise_distance
import multiprocessing as mp

def _neighborhood(A,pwise_dist,cutoff=2.): 
    D = pwise_dist
    mediana = np.average(np.sort(D,axis=0)[1,:])
    neighborWeights = np.where(D<cutoff*mediana,1.,0.)
    neighborN = np.sum(neighborWeights,axis=0)
    
    return neighborN, neighborWeights
            
def _studt(X,Y,sigma2,Gamma,D,pwise_dist):
    Gamma = Gamma
    A = spgamma(0.5*(Gamma+D))
    B = np.sqrt(2.*sigma2) * (Gamma+spgamma(0.5))**(0.5*D) * spgamma(0.5*Gamma)
    C = (1.0 + pwise_dist*(1./(sigma2*Gamma[:,None])))**(0.5*(D+Gamma[:,None]))
    
    return A[:,None]/(B[:,None]*C)
    
def _eqforgamma(Gamma,Gamma_old,p,u,D,CDE_term):
    Gammahalves = 0.5*Gamma
    A = -spdigamma(Gammahalves)
    B = np.log(Gammahalves)
    
    jac = -0.5*sppolygamma(1,Gammahalves)+1./Gammahalves
    Jac = np.diag(jac)
    
    return (1.+A+B+CDE_term), Jac
    
def _eqforalpha(alpha,p,sumPoverN):
    #Eq. (20)', switching indices for x and y to match their first paper. 
    #Added an alpha to the first term inside the \biggl ( in the final line!!
    A_term = np.exp(alpha*sumPoverN)
    B_term = np.sum(sumPoverN * A_term,axis=1) #There's an alpha here or not???
    C_term = np.sum(A_term,axis=1)
    D_term = -B_term/C_term
    
    return np.sum(p*(alpha*sumPoverN+D_term[:,None]))
    
def dsmm(A,B,**kwargs):
    # If multiple A pointsets are passed, run the registrations in parallel
    if len(A.shape)==3:
        pool = mp.Pool(kwargs['cores'])
        P = [pool.apply(_dsmm, args=(A[i],B,True), kwargs=kwargs)
             for i in np.arange(A.shape[0])]
        return P
        
    elif len(A.shape)==2:
        return _dsmm(A,B,**kwargs)

def _dsmm(A,B,returnOnlyP=False,beta=2.0,llambda=1.5,neighbor_cutoff=10.0,
            conv_epsilon=1e-3,eq_tol=1e-2):
    # This is the version running with a single pair of pointsets.
    # This registers Y onto X (Y is what is changed)
    # In wormb.match, I always pass A,B, where B is the refBrain.
    Y = np.copy(A)
    X = np.copy(B)
    
    # Preprocess ("normalize" in Vemuri's language)
    X -= np.average(X,axis=0)
    X /= np.max(np.absolute(X),axis=0)
    X += np.min(X,axis=0)
    Y -= np.average(Y,axis=0)
    Y /= np.max(np.absolute(Y),axis=0)
    Y += np.min(Y,axis=0)
    
    N = X.shape[0]
    M = Y.shape[0]
    D = X.shape[1]
    
    # Init parameters
    pwise_dist = pairwise_distance(X,Y,squared=True).T
    beta2 = beta**2
    w = np.ones((M,N))*(1./M)
    Gamma = np.ones(M)*5.
    Gamma_old = np.copy(Gamma)
    sigma2 = np.sum(pwise_dist)/(D*M*N)
    pwise_distYY = pairwise_distance(Y,Y,squared=True)
    G = np.exp(-pwise_distYY/(2.*beta2))
    Identity = np.diag(np.ones(M))

    # Allocate arrays
    F_t = np.empty((M,N))
    p = np.empty((M,N))
    u = np.empty((M,N))
    hatP = np.empty((M,N))
    hatPI = np.empty((M,M))
    hatPIG = np.empty((M,M))
    hatPX = np.empty((M,D))
    hatPIY = np.empty((M,D))

    #Additional stuff from new paper
    sumPoverN = np.empty((M,N)) 
    alpha=1.
    neighborN, neighborWeights = _neighborhood(Y,pwise_distYY,neighbor_cutoff)
    
    # Convergence
    relerr = 1000.
    regerror = np.sum(pairwise_distance(X,Y,squared=True))

    i = 0
    ##### array[y,x]
    while relerr > conv_epsilon:
        #Step3 (Eq. (5))
        F_t[:] = _studt(X,Y,sigma2,Gamma,D,pwise_dist)

        #Step3:E-Step
        #Eq. (17)'
        wF_t = w*F_t
        p[:] = wF_t / np.sum(wF_t,axis=0)[None,:]
        #p = np.absolute(p)
        
        #Eq. (16) and (21)'
        u[:] = (Gamma[:,None] + D) / (Gamma[:,None] + pwise_dist/sigma2)
        #u = np.absolute(u)
        
        #Eq. (20)'
        neighborN, neighborWeights = _neighborhood(Y,pwise_distYY,neighbor_cutoff)
        sumPoverN = np.sum(neighborWeights[...,None]*p[None,...],axis=1)/neighborN[:,None]  
        Result = sproot(_eqforalpha,x0=alpha,args=(p,sumPoverN),method="hybr",tol=eq_tol)
        alpha = Result['x'][0]

        #Step4:M-Step
        # Eq. (18)'
        expAlphaSumPoverN = np.exp(alpha*sumPoverN)
        w[:] = expAlphaSumPoverN/np.sum(expAlphaSumPoverN,axis=0)[None,:]
        
        #Eq. (23)
        Gamma_old = np.copy(Gamma)
        # C,D,E_terms were terms in the function _eqforgamma, but since they 
        # never change, I calculate them just once out here. 
        Gammaoldpdhalves = np.absolute(0.5*(Gamma_old+D))
        C_term = np.sum(p*(np.log(u)-u),axis=1)/np.sum(p,axis=1)
        D_term = spdigamma(Gammaoldpdhalves)
        E_term = -np.log(Gammaoldpdhalves)
        CDE_term = C_term + D_term + E_term
        Result = sproot(_eqforgamma,x0=Gamma_old,args=(Gamma_old,p,u,D,CDE_term),method="hybr",tol=eq_tol,jac=True)
        Gamma = Result['x']

        #Eq. (26)
        hatP[:] = p*u
        hatPI[:] = np.diag(np.dot(hatP,np.ones(N)))
        G[:] = np.exp(-0.5/beta2*pwise_distYY.T) 
        hatPIG[:] = np.dot(hatPI,G)
        hatPX[:] = np.dot(hatP,X)
        hatPIY[:] = np.dot(hatPI,Y)
        A = np.linalg.inv(hatPIG+llambda*sigma2*Identity)
        B = hatPX - hatPIY
        W = np.dot(A,B)
        
        #Step5 moved here to optimize
        Y += np.dot(G,W)
        pwise_dist = pairwise_distance(X,Y,squared=True).T
        pwise_distYY = pairwise_distance(Y,Y,squared=True)

        #Back to Step4 
        #Eq. (27)
        AA = np.sum(p*u*np.sum((X[None,:]-Y[:,None])**2,axis=-1))
        BB = D*np.sum(p*u) # or just p as in the old paper?
        sigma2 = AA*(1./BB)
        
        #Step6
        regerror_old = regerror
        regerror = np.sum(pwise_dist)
        relerr = np.absolute((regerror-regerror_old)/regerror_old)
        i += 1
    
    if returnOnlyP:
        return p
    else:    
        return Y, X, p
