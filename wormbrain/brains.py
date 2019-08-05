import numpy as np
import mistofrutta.struct.irrarray as irrarray
from copy import deepcopy as deepcopy

class Brains:
    '''
    Container for neuron coordinates. It relies on the arrays with "irregular"
    strides from the repository github.org/francescorandi/mistofrutta, which
    are just a shorter notation for specific types of slices of numpy arrays.
    '''
    def __init__(self, coord, volFrame0, zOfFrame=None, properties={}, 
                 stabilize_z=True, coord_ordering='yx'):

        coordZYX, self.nInVolume = self._conv_coord_2d_to_3d(coord, 
                                        volFrame0, dtype=int)
        
        self.coord = irrarray(coordZYX, self.nInVolume, strideNames=["vol"])
        self.volFrame0 = volFrame0
        
        if zOfFrame != None: self.zOfFrame = zOfFrame
        
        if len(properties.keys())!=0:
            self.curvature = properties['curvature']
            self.curvature = irrarray(self.curvature, self.nInVolume, 
                                        strideNames=["vol"])
            self.boxIndices = properties['boxIndices']
            self.boxNPlane = properties['boxNPlane']
        
            if stabilize_z:
                self.coord = self._stabilize_z(self.coord, 
                    self.curvature,
                    nPlane=self.boxNPlane, boxIndices=self.boxIndices,
                    method="xyMaxCurvature")
                    
                self.coord = np.rint(self.coord).astype(int)
                
    def __getitem__(self, i):
        '''
        Allow for direct indexing of the class to access the coordinates.
        '''
        return self.coord.__getitem__(i)
        
    def __setitem__(self, i, value):
        '''
        Allow for direct indexing of the class to write in the coordinates.
        '''
        self.coord.__setitem__(i,value)
        
    def __call__(self, *args, **kwargs):
        '''
        Upon call, use the __call__ method of the coordinates irrarray.
        '''
        return self.coord.__call__(*args, **kwargs)
        
    def copy(self):
        return deepcopy(self)
        
    def trueCoords(self, vol, coord_ordering='zyx'):
        '''
        Return the coordinates replacing z with its actual values, from 
        zOfFrame.
        '''
        if type(vol)!=list: vol = [vol]
        # Get the neurons in the requested volumes
        trueCoords = self.coord(vol=vol, dtype=np.float)
        intCoords = self.coord(vol=vol)
        
        L = len(vol)
        for l in np.arange(L):
            trueCoords[l][:,0] = self.zOfFrame[vol[l]][intCoords[l][:,0]]
        
            # Ordering stuff
            if coord_ordering=="xyz": 
                trueCoords[l] = trueCoords[l][:,::-1]
                
        if len(trueCoords)==1: trueCoords = trueCoords[0]
                
        return trueCoords
        
    @staticmethod
    def _conv_coord_2d_to_3d(coord_2d, volFrame0, zOfFrame=[], dz=1, 
            dtype=np.float, coord_2d_ordering='yx', coord_3d_ordering='zyx'):
        '''
        Converts coordinates from a list of np.array([[y,x],]) for one frame to
        a list of np.array([[z,y,x],]) for each volume, with the corresponding
        number of neurons in each volume.
        
        Parameters
        ----------
        coord_2d: list of numpy arrays
            coord_2d[i][j,n] gives the coordinate n of neuron j in frame i
        nInFrame: numpy array
            nInFrame[i] gives the number of neurons in frame i
        volFrame0: numpy array
            volFrame0[l] gives the first frame of volume l. As last element, it
            must contain also the last+1 frame of the last volume.
        zOfFrame: numpy array
            zOfFrame[i] gives the z coordinate of frame i
        dz: scalar
            step to build the z coordinates without passing zOfFrame
        dtype: data type
            This function will produce an output array of this type. Therefore,
            an integer type and a float dz will produce floored z coordinates. And zOfFrame too.
        coord_2d_ordering: string
            Specifies the ordering of the coordinates inside the input array.
            Default is 'yx' (indexing order).
        coord_3d_ordering: string
            Specifies the ordering of the coordinates inside the returned array.
            Default is 'zyx', for indexing. For plotting, use 'xyz'.
            
        Returns
        -------
        coord_3d: numpy array
            coord_3d[j, n] gives the coordinate n of neuron j.
        nInVolume: numpy array
            nInVolume[l] gives the number of neurons in volume l.
        '''

        # Make sure nInFrame is an integer and can be used as an index.
        #nInFrame = nInFrame.astype(int, copy=False)
        nInFrame = np.array([c.shape[0] for c in coord_2d])
        
        # Initialize the lists containing the neurons coordinates.
        nTotal = np.sum(nInFrame)
        
        coord_3d = np.zeros((nTotal,3))
        nInVolume = []
        L = len(volFrame0)-1
        
        #For each volume
        g = 0
        for l in np.arange(L):
            # First and last+1 frames of current volume
            firstframe = volFrame0[l]
            lastframeplus1 = volFrame0[l+1]
            
            # Add an entry to the array listing the number of neurons in each
            # volume.
            nInVolume.append(np.sum(nInFrame[firstframe:lastframeplus1]))
            
            # Initialize an array to be populated with the coordinates of the 
            # neurons in the current volume. 
            NeuronInVolume = np.zeros((nInVolume[-1],3),dtype=dtype)
            
            q = 0
            # For each frame in the volume
            for i in np.arange(firstframe, lastframeplus1):
                # If no zOfFrame was passed, used dz to build it. If zOfFrame is
                # present, copy the value for this frame in an array the size
                # of the number of neurons in this frame.
                if len(zOfFrame)==0:
                    Z = np.ones(nInFrame[i],dtype=dtype)*(i-firstframe)*dz
                else:
                    Z = np.ones(nInFrame[i])*zOfFrame[l][i-firstframe]
                    
                # Depending on the specified ordering, extract X and Y from 
                # coord_2d
                if coord_2d_ordering=='xy':
                    X,Y = coord_2d[i].T
                else:
                    Y,X = coord_2d[i].T
                    
                # Depending on the specified ordering, combine the X, Y, and Z
                # coordinates.
                if coord_3d_ordering=='xyz':
                    tmp = np.array([X,Y,Z]).T
                elif coord_3d_ordering=='zyx':
                    tmp = np.array([Z,Y,X]).T
                NeuronInVolume[q:q+nInFrame[i]] = tmp
                 
                q += nInFrame[i]
                 
            #coord_3d.append(NeuronInVolume)
            coord_3d[g:g+q] = NeuronInVolume
            g += q
            
        return coord_3d, np.array(nInVolume)
        
    @staticmethod
    def _stabilize_z(coord, curvature, nPlane=7, boxIndices=
        [np.arange(1),np.arange(1,6),np.arange(6,19),np.arange(19,32),
        np.arange(32,45),np.arange(45,50),np.arange(50,51)], 
        coord_3d_ordering="zyx", method=""):
        '''
        Stabilizes the z position of the neuron using the local curvature around
        that point. This helps in obtaining less fluctuating results when the
        neurons were found in 2D with brute-force check along z, i.e. with 
        smoothing in xy but not in z, as it is done in the neuronsegmentation 
        module. The z position of each neuron is shifted by the average of 
        position in the box around the neuron weighted by the curvature.
        
        Parameters
        ----------
        coord: numpy array
            coord_3d[j, n] gives the coordinate n of neuron j (counting from 
            neuron 0 in volume 0).
            The z coordinate must be the index of the corresponding frame, and
            not the actual coordinate.
        curvature: numpy arrays
            curvatureVSplit[j, m] gives the curvature at point m in the box
            (described by nPlane and boxIndices) around neuron j.
        nPlane: integer scalar
            number of planes spanned by the box around each neuron
        boxIndices: list of numpy arrays
            boxIndices[pl] gives the indices of each curvatureVSplit[l][j] that
            reside in plane pl.
        coord_3d_ordering: string
            Specifies the ordering of the coordinates inside the coordinate array.
            Default is 'zyx', (indexing ordering). The output array has the same
            odering as the input array.
        method: string (optional)
            If "xyMaxCurvature", the weighting is performed with the maximum
            curvature in each plane. Otherwise, only with the curvatures directly
            above and below the neuron position.
            
        Returns
        -------
        coord_3d_out: numpy array
            coord_3d_out[j, n] gives the coordinate n of the stabilized neuron j 
        '''
        # Determine the index of the z coordinate in the input and output arrays
        z_indices = {"zyx":0,"xyz":2}
        z_index = z_indices[coord_3d_ordering]
        
        # Build z range around 0 for specified nPlanes
        z = np.arange(-(nPlane//2),nPlane//2+1,dtype=np.float)
        
        curv = np.zeros((coord.shape[0],nPlane))
        if method=="xyMaxCurvature":
            for pl in np.arange(nPlane):
                c = np.max(curvature[:,boxIndices[pl]],axis=1)
                curv[:,pl] = c
        else:
            centralIndices = np.zeros(nPlane)
            for pl in np.arange(nPlane):
                sh0 = boxIndices[pl].shape[0]
                centralIndices[pl] = boxIndices[pl][sh0//2]
            curv = curvature[:,centralIndices] #look just along z
        
        coord[:,z_index] += np.sum(z*curv,axis=1)/np.sum(curv,axis=1)
        
        return coord
