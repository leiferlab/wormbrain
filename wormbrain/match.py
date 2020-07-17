import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import wormbrain as wormb
import pkg_resources

filename_matches = "matches.txt"

def pairwise_distance(A,B,returnAll=False,squared=False,thresholdDz=0.0):
    '''
    Calculates the pairwise distance between points belonging to two sets of
    points.

    Parameters
    ----------
    A, B: np.array
        Sets of points between witch to calculate the distance. Indexes are
        [point_index, coordinate].
    returnAll: bool (optional)
        If True, the function returns both the distance and the vector 
        difference between the points in A and B. Default: False
    squared: bool (optional)
        If True, the square of the distances are returned, instead of the 
        distances. Default: False.
    thresholdDz: float (optional)
        If different from 0.0, the function sets to 0 all the z components
        of the vectorial difference between A and B that are smaller than this
        value.

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
    A, B: numpy array
        Sets of points to match. The function will calculate matches B->A.
        Indexes are [point_index, coordinate].
    method: string (optional)
        Method to match the neurons, after the registration between the two
        point sets. Default: "nearest".
    registration: string (optional)
        Method to register the two point sets. Default: "None".
    **kwargs:
        Other parameters to be passed to the registration function.

    Returns
    -------
    Match: np.array
        Indexes of points in A matching points in B.
        index_point_in_A = Match[index_point_in_B]

    '''

    if registration=='centroid':
        A, B = wormb.reg.centroid(A,B,**kwargs)
    elif registration=='displacement':
        A, B = wormb.reg.displacement(A,B,**kwargs)
    #elif registration=='tps':
    #    A, B = wormb.reg.tps(A,B,**kwargs)
    elif registration=="dsmm":
        try:
            fullpy = kwargs.pop("fullpy")
        except:
            fullpy = False
            
        if fullpy:
            B, A, p, Match = wormb.reg._dsmm_fullpy(B,A,returnAll=True,**kwargs)
        else:
            B, A, p, Match = wormb.reg.dsmmc(B,A,returnAll=True,**kwargs)
        return Match

    if method=='nearest':
        Match = _match_nearest(A,B,**kwargs)
    return Match
    
def invert_matches(matches, N):
    '''Transforms the matches obtained matching points in B to points in A to
    the matches of points in A to points in B.
    
    Parameters
    ----------
    matches: numpy array of int
        The matches to be inverted. Obtained, e.g., with match(A,B,...).
        Contains the matches of points in B to points in A.
    N: int
        Number of elements in A.
        
    Returns
    -------
    inverted matches: numpy array of int
        The inverted matches, i.e. of points of A to points of B.
    '''
    
    inverted_matches = -1*np.ones(N, dtype=np.int)
    for i in np.arange(N):
        a = np.where(matches==i)[0]
        if len(a) >0:
            inverted_matches[i] = a[0]
    
    return inverted_matches



def _match_nearest(A, B, **kwargs): #TODO implement it on multiple As
    '''Matches the neurons in A to the neurons in B based on a nearest-neighbor
    criterion.
    
    Parameters
    ----------
    A, B: numpy array
        The two point sets
        
    Returns
    -------
    Match: numpy array
        The array containing the matches.
    '''
    
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

def save_matches(MMatch, parameters, folder, filename=""):
    '''Saves the matches to file, using the numpy savetxt function with the
    addition of a header containing json-serialized information about the 
    matching. The function will add the current version of the wormbrain module
    in the header.
    
    Parameters
    ----------
    MMatch: numpy array
        Array containing the matches
    parameters: dictionary
        Dictionary to be json-serialized and stored in the header.
    folder: string
        Destination folder.
    filename: string (optional)
        Destination filename. Default: "", that is translated in the default
        filename for the module.
    '''
    
    if folder[-1]!="/": folder+="/"
    if filename=="": filename = filename_matches
    
    parameters['version'] = pkg_resources.get_distribution("wormbrain").version

    headerInfo = json.dumps(parameters)
    np.savetxt(folder+filename,MMatch,header=headerInfo)

def load_matches(folder, filename=""):
    '''Loads the matches from file.
    
    Parameters
    ----------
    folder: string
        Folder containing the file.
    filename: string (optional)
        Name of the file. Default: "", that is translated to the default 
        filename for the module.
    
    Returns
    -------
    MMatch: numpy array
        Array containing the matches.
    parameters: dictionary
        Dictionary contained in the header, with the information about the 
        matching
    '''
    
    if folder[-1]!="/": folder+="/"
    if filename=="": filename = filename_matches
    
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        parameters = json.loads(l[2:])
    except:
        parameters = {}

    MMatch = np.loadtxt(folder+filename_matches)

    return MMatch, parameters
    
def load_match_parameters(folder, filename=""):
    '''Loads the matching parameters from file, only using its header. This
    function does not load the matches themselves. Use load_matches() for that.
    
    Parameters
    ----------
    folder: string
        Folder containing the file.
    filename: string (optional)
        Name of the file. Default: "", that is translated to the default 
        filename for the module.
    
    Returns
    -------
    parameters: dictionary
        Dictionary contained in the header, with the information about the 
        matching
    '''
    if folder[-1]!="/": folder+="/"
    if filename=="": filename = filename_matches
    
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        parameters = json.loads(l[2:])
    except:
        parameters = {}

    return parameters


def plot_matches(A, B, Match, mode='3d',plotNow=True,**kwargs):
    '''Plots the two point sets with lines representing the matches, in 2D or 
    3D. The function redirects the call to _plot_matches_3d() or _2d(). If 
    matplotlib has already a figure, the function uses the one after the one
    currently in use.
    
    Parameters
    ----------
    A, B: numpy arrays
        The two point sets.
    Match: numpy array
        The array containing the matches.
    mode: string (optional)
        Possible values: "3d" or "2d". Default: "3d".
    plotNow: boolean (optional)
        If True, the plot is displayed immediately. If False, figure and axis
        will be returned, to be displayed later in the script. Default: True.
    **kwargs: 
        Any other parameter to be passed to _plot_matches_3d or _2d.
        
    Returns
    -------
    fig: matplotlib figure 
        Returned if plotNow is False.
    ax: matplotlib axis
        Returned if plotNow is False.
    
    '''
    if mode=='3d':
        fig, ax = _plot_matches_3d(A, B, Match, **kwargs)
    if mode=='2d':
        fig, ax = _plot_matches_2d(A, B, Match, **kwargs)

    if plotNow==True:
        plt.show()
        return
    else:
        return fig, ax


def _plot_matches_3d(A, B, Match,**kwargs):
    '''Plot matches in a 3D plot. If matplotlib has already a figure, the 
    function uses the one after the one currently in use.
    
    Parameters
    ----------
    A, B: numpy arrays
        The arrays containing the point sets.
    Match: numpy array
        The matches.
    
    Returns
    -------
    fig: matplotlib figure 
    ax: matplotlib axis
    '''
    
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
    '''Plot matches in a 2D plot. If matplotlib has already a figure, the 
    function uses the one after the one currently in use.
    
    Parameters
    ----------
    A, B: numpy arrays
        The arrays containing the point sets.
    Match: numpy array
        The matches.
    
    Returns
    -------
    fig: matplotlib figure 
    ax: matplotlib axis
    '''
    
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
