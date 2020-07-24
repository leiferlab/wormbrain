import numpy as np
from wormbrain.match import pairwise_distance

def centroid(A,B,**kwargs):
    '''Simple registration in which the pointset B is shifted so that its 
    centroid overlaps with the centroid of A.
    
    Parameters
    ----------
    A, B: numpy arrays
        The two point sets.
        
    Returns
    -------
    A, B: numpy arrays
        The two point sets, with B shifted.
    '''
    
    centroidAxes = kwargs['centroidAxes'] #tuple
    centroidA = np.average(A,axis=0)
    centroidB = np.average(B,axis=0)
    centroidAB = centroidA - centroidB
    
    for axis in centroidAxes:
        B[:,axis] += centroidAB[axis]
        
    return A,B
    
def displacement(A,B,**kwargs):
    '''Simple registration in which the pointset B is shifted using the median
    or the average of the nearest-neighbor distances.
    
    Parameters
    ----------
    A, B: numpy arrays
        The two point sets.
    displacementMethod: string (optional)
        If "median", the median of the nearest-neighbor distances is calculate.
        Otherwise, the average is used. Default: "average".
        
    Returns
    -------
    A, B: numpy arrays
        The two point sets, with B shifted.
    '''
    
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
