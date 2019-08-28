import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pairwise_distance(A,B,returnAll=False,squared=False,thresholdDz=0.0):
    '''
    Calculates the pairwise distance between points belonging to two sets of 
    points.
    
    Parameters
    ----------
    A, B: np.array
        Sets of points between witch to calculate the distance. Indexes are
        [point_index, coordinate].
        
    Returns
    -------
    D: np.array
        Matrix of the distances between each point in A and each point in B.
        Indexes are D[index_point_in_A, index_point_in_B].
        
    '''
    Dv = A[:,None]-B[None,:]

    if thresholdDz!=0.0:
        Dv[:,2,:] = 0.5*(np.sign(np.absolute(Dv[:,2,:])-thresholdDz)+1.0)*Dv[:,2,:]
        
    D = np.sum(np.power(Dv,2),axis=-1) # [a,b]
    if not squared:
        D = np.sqrt(D)
        
    if returnAll==True:
        return D, Dv
    else:
        return D
    
    
def match(A, B, method='nearest', registration='None', **kwargs):
    '''
    Computes the array giving the matching of points in B to points in A.
    
    Parameters
    ----------
    A, B: np.array
        Sets of points to match. The function will calculate matches B->A.
        Indexes are [point_index, coordinate].
        
    Returns
    -------
    Match: np.array
        Indexes of points in A matching points in B.
        index_point_in_A = Match[index_point_in_B]
    
    '''
    
    if registration=='centroid':
        A, B = _register_centroid(A,B,**kwargs)
    elif registration=='displacement':
        A, B = _register_displacement(A,B,**kwargs)
    #elif registration=='tps':
    #    A, B = _register_tps(A,B,**kwargs)
    elif registration=="dsmm":
        A, B = wormb.reg._dsmm(A,B,**kwargs) #TODO implement parallelization inside it
    
    if method=='nearest':
        Match = _match_nearest(A,B,**kwargs)
    return Match
    
    
    
def _match_nearest(A, B, **kwargs): #TODO implement it on multiple As
    # Calculate pairwise distance between every pair of points in (B, A).T
    # Pass thresholdDz if it is in kwargs
    if "thresholdDz" in kwargs.keys():
        DD = pairwise_distance(A, B, thresholdDz=kwargs["thresholdDz"])
    else:
        DD = pairwise_distance(A, B) # [a, b]
    # Find closest match for each point
    # For each B find closest A
    MatchAll = np.argsort(DD,axis=0) #0 or 1 depends on whether you want to match the A to the model or viceversa
    Match = MatchAll[0]
    distanceThreshold = kwargs['distanceThreshold']
    DDth = distanceThreshold*np.median(np.min(DD,axis=0))
    Match[np.where(np.min(DD,axis=0)>DDth)] *= -10
    
    # Look for double matches
    unique, counts = np.unique(Match, return_counts=True)
    doubleMatch = np.where((counts!=1)*(unique>=0))[0]
    
    # Keep only closest match
    for dm in doubleMatch:
        if unique[dm]>0:
            # Bs matched to the same As
            pts = np.where(Match==unique[dm])[0]
            # Don't change the closest match
            pts = np.delete(pts, np.argmin(DD[unique[dm],pts]))
            # Assign the second closest match, if it is not already assigned. Should be recursive to get to the third closest and so on, with cutoff on the distance
            remainingPtsIndexes = np.argsort(DD[unique[dm],pts])
            for rpi in remainingPtsIndexes:
                pt = pts[rpi] # the next minimum
                jj = 1
                run = True
                while run:
                    if DD[MatchAll[jj,pt],pt] < DDth:
                        if not np.isin(MatchAll[jj,pt],Match):
                            Match[pt] = MatchAll[jj,pt]
                            run = False
                    else:
                        Match[pt] *= -10
                        run = False
                    jj += 1
                    
                    if jj > MatchAll.shape[0] - 2: run = False
            
            #Match[pts] *= -10
                #Match[pt] = np.where( (not np.isin(MatchAll[1,pt],Match)) and (DD[MatchAll[1,pt],pt] < DDth), MatchAll[1,pt], -10*Match[pt])
                
    return Match
    
def _register_centroid(A,B,**kwargs):
    centroidAxes = kwargs['centroidAxes'] #tuple
    centroidA = np.average(A,axis=0)
    centroidB = np.average(B,axis=0)
    centroidAB = centroidA - centroidB
    
    for axis in centroidAxes:
        B[:,axis] += centroidAB[axis]
        
    return A,B
    
def _register_displacement(A,B,**kwargs):
    # get both distance and vectors
    DD, Dv = pairwise_distance(A, B, returnAll=True)
    
    # extract the vectors of the closest matches
    Match = np.argsort(DD, axis=0)[0]
    Dvp = Dv[Match,:,np.arange(len(Match))]
    
    if kwargs['displacementMethod']=="median":
        Dvshift = np.median(Dvp, axis=(0))
    else:
        Dvshift = np.average(Dvp, axis=(0))
    
    for i in np.arange(Dvshift.shape[0]):
        B[:,i] += Dvshift[i]
    
    return A, B
    

def plot_matches(A, B, Match, mode='3d',plotNow=True,**kwargs):
    
    if mode=='3d':
        fig, ax = _plot_matches_3d(A, B, Match, **kwargs)
    if mode=='2d':
        fig, ax = _plot_matches_2d(A, B, Match, **kwargs)
        
    if plotNow==True:
        plt.show()
        return
    else:
        return fig, ax
        
        
def _plot_matches_3d(A, B, Match):
    cfn = plt.gcf().number
    if len(plt.gcf().axes)!=0: cfn += 1
    
    showAll=True
    if 'showAll' in kwargs: showAll=kwargs['showAll']
    
    fig = plt.figure(cfn)
    ax = fig.add_subplot(111,projection='3d')
    
    ax.scatter(A.T[0],A.T[1],A.T[2],'o',c='green')
    ax.scatter(B.T[0],B.T[1],B.T[2],'o',c='red',s=2)
    I = len(Match)
    for i in np.arange(I):
        j = Match[i]
        if j>0:
            ax.plot((A[j,0],B[i,0]),(A[j,1],B[i,1]),(A[j,2],B[i,2]),'k-')
        else:
            j = -j//10
            ax.scatter(B[i,0],B[i,1],B[i,2],'*',c='blue')
            if showAll:
                ax.plot((A[j,0],B[i,0]),(A[j,1],B[i,1]),(A[j,2],B[i,2]),'--',c='orange')
            
    return fig, ax
    

def _plot_matches_2d(A, B, Match, **kwargs):
    cfn = plt.gcf().number
    if len(plt.gcf().axes)!=0: cfn += 1
    
    showAll=True
    if 'showAll' in kwargs: showAll=kwargs['showAll']
    
    fig = plt.figure(cfn)
    ax = fig.add_subplot(111)
    p = 0 # x in plot
    q = 1 # y in plot
    r = 2

    ax.plot(A.T[p],A.T[q],'og')#,markersize=3)
    ax.plot(B.T[p],B.T[q],'or',markersize=2)

    I = len(Match)
    for i in np.arange(I):
        j = Match[i]
        if j>0:
            ax.plot((A[j,p],B[i,p]),(A[j,q],B[i,q]),'k-')
        else:
            j = -j//10
            ax.plot(B[i,p],B[i,q],'*b')
            if showAll:
                ax.plot((A[j,p],B[i,p]),(A[j,q],B[i,q]),'--',c='orange')
            
    return fig, ax
