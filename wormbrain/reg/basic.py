import numpy as np
from wormbrain.match import pairwise_distance

def centroid(A,B,**kwargs):
    centroidAxes = kwargs['centroidAxes'] #tuple
    centroidA = np.average(A,axis=0)
    centroidB = np.average(B,axis=0)
    centroidAB = centroidA - centroidB
    
    for axis in centroidAxes:
        B[:,axis] += centroidAB[axis]
        
    return A,B
    
def displacement(A,B,**kwargs):
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
