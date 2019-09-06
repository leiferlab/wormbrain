import numpy as np
from scipy.special import gamma as spgamma
from scipy.special import digamma as spdigamma
from scipy.special import polygamma as sppolygamma
from scipy.special import bernoulli as spbernoulli
from scipy.optimize import root as sproot
from wormbrain.match import pairwise_distance
import mistofrutta as mf
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

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
    mf.approx.exp(alpha*sumPoverN,np.prod(sumPoverN.shape),A_term,0)
    #A_term = np.exp(alpha*sumPoverN) # TODO approximate this exponential
    B_term = np.sum(sumPoverN * A_term,axis=1) #There's an alpha here or not???
    C_term = np.sum(A_term,axis=1)
    D_term = -B_term/C_term
    
    return np.sum(p*(alpha*sumPoverN+D_term[:,None]))

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

def _dsmm(Y,X,returnOnlyP=False,beta=2.0,llambda=1.5,neighbor_cutoff=10.0,
            conv_epsilon=1e-3,eq_tol=1e-2,Ashape=(1,1),Bshape=(1,1)):
    # This is the version running with a single pair of pointsets.
    # This registers Y onto X (Y is what is changed)
    # In wormb.match, I always pass A,B, where B is the refBrain.
    tm1 = time.time()
    #Y = np.copy(A)
    #X = np.copy(B)
    
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
    pwise_dist = pairwise_distance(Y,X,squared=True)
    mf.approx.pwise_dist2(Y,X,M,N,D,pwise_dist)
    beta2 = beta**2
    w = np.ones((M,N))*(1./M/N)
    Gamma = np.ones(M)*3.
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
    #D_term = np.empty(M)
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
    
    tm2 = time.time()
    i = 0
    ##### array[y,x]
    while relerr > conv_epsilon:
        t0 = time.time()
        #Step3 (Eq. (5))
        #F_t[:] = _studt(sigma2,Gamma,D,pwise_dist)
        
        mf.approx.studt(pwise_dist,M,N,sigma2,Gamma,D,F_t)

        #Step3:E-Step
        #Eq. (17)'
        '''
        wF_t = w*F_t
        p[:] = wF_t / np.sum(wF_t,axis=0)[None,:]
        '''
        mf.approx.step1(w,F_t,wF_t,wF_t_sum,p,M,N)
        #p = np.absolute(p)
        
        #Eq. (16) and (21)'
        '''
        u[:] = (Gamma[:,None] + D) / (Gamma[:,None] + pwise_dist/sigma2)
        '''
        mf.approx.step2(u,Gamma,pwise_dist,sigma2,M,N,D)
        #u = np.absolute(u)

        t1 = time.time()
        #Eq. (20)'
        #neighborN, neighborWeights = _neighborhood(pwise_distYY,neighbor_cutoff)
        #sumPoverN = np.sum(neighborWeights[...,None]*p[None,...],axis=1)/neighborN[:,None] ##############TODO this takes a long time! 2.5 ms! It's because I augment the dimensionality so much.
        t2b = time.time()
        mf.approx.getsumPoverN(pwise_distYY, N, N, neighbor_cutoff, p, sumPoverN) #yay!
        t2c = time.time()
        
        #Result = sproot(_eqforalpha,x0=alpha,args=(p,sumPoverN),method="hybr",tol=eq_tol)
        #alpha = Result['x'][0]
        Alpha = np.array([alpha])
        mf.approx.solve_foralpha(p,M,N,sumPoverN,Alpha)
        alpha = Alpha[0]
        
        t2 = time.time()
        #Step4:M-Step
        # Eq. (18)'
        '''
        expAlphaSumPoverN = np.exp(alpha*sumPoverN)
        w[:] = expAlphaSumPoverN*(1./np.sum(expAlphaSumPoverN,axis=0)[None,:])
        '''
        mf.approx.step3(sumPoverN,expAlphaSumPoverN,w,alpha,M,N)
        t3 = time.time()
        
        #Eq. (23)
        #Gamma_old = np.copy(Gamma)
        # C,D,E_terms were terms in the function _eqforgamma, but since they 
        # never change, I calculate them just once out here. 
        '''
        Gammaoldpdhalves = np.absolute(0.5*(Gamma_old+D))
        C_term = np.sum(p*(np.log(u)-u),axis=1)/np.sum(p,axis=1)
        D_term = spdigamma(Gammaoldpdhalves)
        E_term = -np.log(Gammaoldpdhalves)
        CDE_term = C_term + D_term + E_term'''
        mf.approx.step4(Gamma,p,u,CDE_term,M,N,D)
        #print("CDE_term")
        #print(CDE_term[0:4])
        #Result = sproot(_eqforgamma,x0=Gamma_old,args=(CDE_term),method="hybr",tol=eq_tol,jac=True,options={'col_deriv':1})
        #Gamma = Result['x']
        mf.approx.solve_forgamma(CDE_term,len(CDE_term),Gamma)
         

        t4 = time.time()
        #Eq. (26)
        # TODO hatP[:] = p*u
        #hatPI[:] = np.diag(np.dot(hatP,np.ones(N)))
        #TODO hatPI_diag[:] = np.sum(hatP,axis=1)
        #hatPI[:] = np.diag(hatPI_diag)
        #TODO G[:] = np.exp(-0.5/beta2*pwise_distYY) #TODO ????.T
        t4d = time.time()
        #hatPIG[:] = np.dot(hatPI,G)
        #mf.approx.mfdot(hatPI,G,hatPI.shape[0],hatPI.shape[1],G.shape[1],hatPIG,True) #TODO recheck times. TODO the first matrix is diagonal: the dot product is simpler!
        #mf.approx.mfdot_diag(hatPI_diag,G,G.shape[0],G.shape[1],hatPIG,True)
        #TODO hatPIG[:] = hatPI_diag[:,None]*G
        
        #mf.approx.mfdot(hatP,X,hatP.shape[0],hatP.shape[1],X.shape[1],hatPX,True)
        #hatPIY[:] = np.dot(hatPI,Y)
        #mf.approx.mfdot(hatPI,Y,hatPI.shape[0],hatPI.shape[1],Y.shape[1],hatPIY,True)
        #TODO hatPIY = hatPI_diag[:,None]*Y
        mf.approx.step5(p,u,hatP,hatPI_diag,hatPIG,hatPIY,hatPX,G,W,GW,X,Y,pwise_dist,pwise_distYY,beta2,llambda,sigma2,M,N,D)
        #print("W ")
        #print(W[0])
        #print("Y ")
        #print(Y[0])
         
        #hatPX[:] = np.dot(hatP,X)
        t4e = time.time()
        #TODO A = np.linalg.inv(hatPIG)#TODO FIXME +llambda*sigma2*Identity)
        t4f = time.time()
        #TODO B = hatPX - hatPIY
        #TODO W = np.dot(A,B)
        #mf.approx.mfdot(A,B,A.shape[0],A.shape[1],B.shape[1],W,True)
        #Step5 moved here to optimize
        #mf.approx.mfdot(G,W,G.shape[0],G.shape[1],W.shape[1],GW,True)
        #Y += GW
        #TODO Y += np.dot(G,W)
        t4b = time.time()
        #pwise_dist = pairwise_distance(Y,X,squared=True) 
        #pwise_distYY = pairwise_distance(Y,Y,squared=True)
        #TODO mf.approx.pwise_dist2(Y,X,M,N,D,pwise_dist)
        #TODO mf.approx.pwise_dist2_same(Y,M,D,pwise_distYY)
        t4c = time.time()
        t5 = time.time()
        #Back to Step4 
        #Eq. (27)
        AA = np.sum(hatP*pwise_dist)#np.sum((X[None,:]-Y[:,None])**2,axis=-1))
        BB = D*np.sum(hatP) # or just p as in the old paper?
        sigma2 = AA*(1./BB)
        #print(sigma2)
        t5b = time.time()
        #Step6
        regerror_old = regerror
        regerror = np.sum(pwise_dist)
        relerr = np.absolute((regerror-regerror_old)/regerror_old)
        
        '''if i==1:
            print("py")
            print("F_t")
            print(F_t[0,0:4])
            print("p")
            print(p[0,0:4])
            print("u")
            print(u[0,0:4])
            print("sumPoverN")
            print(sumPoverN[0,0:4])
            print("alpha")
            print(alpha)
            print("CDE")
            print(CDE_term[0:4])  
            print("gamma")
            print(Gamma[0:4])  
            print("Y")
            print(Y[0])
            print("sigma2")
            print(sigma2)
            #quit()'''
        
        i += 1
        if i>1: t6bis = t6
        t6 = time.time()

    print("t-2-t-1",tm2-tm1)   
    print("t1-t0",t1-t0)
    print("t2b-t1",t2b-t1)
    print("t2c-t2b",t2c-t2b)
    print("t2-t2c",t2-t2c)
    print("t2-t1",t2-t1)
    print("t3-t2",t3-t2)
    print("t4-t3",t4-t3)
    print("t4c-t4b",t4c-t4b)
    print("t4d-t4",t4d-t4)
    print("t4e-t4d",t4e-t4d)
    print("t4f-t4e",t4f-t4e)
    print("t5-t4",t5-t4)
    print("t5b-t5",t5b-t5)
    print("t6-t5",t6-t5)
    print("t6-t0",t6-t0)
    try:
        print("t0-t6b",t0-t6bis)
    except:
        pass
    print("i",i)
    print("ratio",(t2-t1+t4-t3)/(t6-t0))
    
    if returnOnlyP:
        return p
    else:    
        return Y, X, p
