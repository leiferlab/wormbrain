import numpy as np
from scipy.special import gamma as spgamma
from scipy.special import digamma as spdigamma
from scipy.special import polygamma as sppolygamma
from scipy.optimize import root as sproot
import wormbrain as wormb

X = np.loadtxt("data/fish_X.txt")#[:-4]
X = np.random.permutation(X)[:40]
Y = np.loadtxt("data/fish_Y.txt")
#Y = np.random.permutation(Y)[:60]

# Preprocess ("normalize" in Vemuri's language)
# Signficantly reduces convergence time
X -= np.average(X,axis=0)
X /= np.max(np.absolute(X),axis=0)
X += np.min(X,axis=0)
Y -= np.average(Y,axis=0)
Y /= np.max(np.absolute(Y),axis=0)
Y += np.min(Y,axis=0)

Y_orig = np.copy(Y)

N = X.shape[0]
M = Y.shape[0]
D = X.shape[1]

def neighborhood(A,pwise_dist,cutoff=2.): 
    #D = pairwise_distance(A,A)
    D = pwise_dist
    mediana = np.average(np.sort(D,axis=0)[1,:])
    neighborWeights = np.where(D<cutoff*mediana,1.,0.)
    neighborN = np.sum(neighborWeights,axis=0)
    
    return neighborN, neighborWeights
            
def studt(X,Y,sigma2,Gamma,D,pwise_dist):
    Gamma = Gamma
    A = spgamma(0.5*(Gamma+D))
    B = np.sqrt(2.*sigma2) * (Gamma+spgamma(0.5))**(0.5*D) * spgamma(0.5*Gamma)
    C = (1.0 + pwise_dist*(1./(sigma2*Gamma[:,None])))**(0.5*(D+Gamma[:,None]))
    
    return A[:,None]/(B[:,None]*C)
    
def eqforgamma(Gamma,Gamma_old,p,u,D,CDE_term):
    Gammahalves = 0.5*Gamma
    A = -spdigamma(Gammahalves)
    B = np.log(Gammahalves)
    
    jac = -0.5*sppolygamma(1,Gammahalves)+1./Gammahalves
    Jac = np.diag(jac)
    
    return (1.+A+B+CDE_term), Jac
    
def eqforalpha(alpha,p,sumPoverN):
    #Eq. (20)', switching indices for x and y to match their first paper. 
    #Added an alpha to the first term inside the \biggl ( in the final line!!
    A_term = np.exp(alpha*sumPoverN)
    B_term = np.sum(sumPoverN * A_term,axis=1) #There's an alpha here or not???
    C_term = np.sum(A_term,axis=1)
    D_term = -B_term/C_term
    
    return np.sum(p*(alpha*sumPoverN+D_term[:,None]))

def _dsmm():
    # Init parameters
    pwise_dist = pairwise_distance(X,Y,squared=True).T
    beta = 2.
    beta2 = beta**2
    w = np.ones((M,N))*(1./M)
    Gamma = np.ones(M)*5.
    Gamma_old = np.copy(Gamma)
    '''extrap_order = 5
    interp_x = np.arange(extrap_order+1)
    extrap_x = np.array([(extrap_order+2)**i for i in np.arange(extrap_order+1)[::-1]]).astype(np.float)
    Gamma_olds = np.empty((extrap_order+1,M))'''
    sigma2 = np.sum(pwise_dist)/(D*M*N)
    G = np.exp(-pairwise_distance(Y,Y,squared=True)/(2.*beta2))
    llambda = 1.5
    Identity = np.diag(np.ones(M))

    # Convergence
    relerr = 1000.
    regerror = np.sum(pairwise_distance(X,Y,squared=True))
    epsilon = 1e-3

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
    neighborN, neighborWeights = neighborhood(Y,2.)

    i = 0
    ##### array[y,x]
    while relerr > epsilon:
        #Step3 (Eq. (5))
        F_t[:] = studt(X,Y,sigma2,Gamma,D,pwise_dist)

        #Step3:E-Step
        #Eq. (17)'
        wF_t = w*F_t
        p[:] = wF_t / np.sum(wF_t,axis=0)[None,:]
        #p = np.absolute(p)
        
        #Eq. (16) and (21)'
        u[:] = (Gamma[:,None] + D) / (Gamma[:,None] + pwise_dist/sigma2)
        #u = np.absolute(u)
        
        #Eq. (20)'
        neighborN, neighborWeights = neighborhood(Y,10.)
        sumPoverN = np.sum(neighborWeights[...,None]*p[None,...],axis=1)/neighborN[:,None]  
        Result = sproot(eqforalpha,x0=alpha,args=(p,sumPoverN),method="hybr",tol=1e-2)
        alpha = Result['x'][0]

        #Step4:M-Step
        # Eq. (18)'
        expAlphaSumPoverN = np.exp(alpha*sumPoverN)
        w[:] = expAlphaSumPoverN/np.sum(expAlphaSumPoverN,axis=0)[None,:]
        
        #Eq. (23) TODO extrapolate the Gamma!!!! 
        Gamma_old = np.copy(Gamma)
        '''Gamma_olds[0:-1] = Gamma_olds[1:]
        Gamma_olds[-1] = np.absolute(Gamma)
        if i<extrap_order:
            Gamma_guess = Gamma_old
        if i>extrap_order+10:
            #print(extrap_x)
            #quit()
            polyfitP = np.polyfit(interp_x,Gamma_olds,extrap_order)
            Gamma_guess = np.sum(polyfitP*extrap_x[:,None],axis=0)#Gamma_guess*0.9+0.1*
            #print(Gamma_guess[0])
            #print(Gamma_old[0])
            #plt.plot(Gamma_guess,'r')
            #plt.plot(Gamma_old,'g')
            #plt.show()
            #quit()
            #print(np.average(np.power(Gamma_guess-Gamma_old,2)))'''
        # C,D,E_terms were terms in the function eqforgamma, but since they never 
        # change, I calculate it just once out here. 
        Gammaoldpdhalves = np.absolute(0.5*(Gamma_old+D))
        C_term = np.sum(p*(np.log(u)-u),axis=1)/np.sum(p,axis=1)
        D_term = spdigamma(Gammaoldpdhalves)
        E_term = -np.log(Gammaoldpdhalves)
        CDE_term = C_term + D_term + E_term
        te = time.time()
        Result = sproot(eqforgamma,x0=Gamma_old,args=(Gamma_old,p,u,D,CDE_term),method="hybr",tol=1e-2,jac=True)
        Gamma = Result['x']

        #Eq. (26)
        hatP[:] = p*u
        hatPI[:] = np.diag(np.dot(hatP,np.ones(N)))
        G[:] = np.exp(-0.5/beta2*pairwise_distance(Y,Y,squared=True).T) # TODO don't calculate pwise_distYY twice!
        hatPIG[:] = np.dot(hatPI,G)
        hatPX[:] = np.dot(hatP,X)
        hatPIY[:] = np.dot(hatPI,Y)
        A = np.linalg.inv(hatPIG+llambda*sigma2*Identity)
        B = hatPX - hatPIY
        W = np.dot(A,B)
        
        #Step5 moved here to optimize
        Y += np.dot(G,W)
        pwise_dist = pairwise_distance(X,Y,squared=True).T

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
