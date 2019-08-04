import numpy as np

class neurons:
    '''
    Basic class containing coordinates of possibly many brains stacked on top
    of each other.
    '''
    
    def __init__(self, coord, zOfFrame, volFrame0, properties={}, 
                 stabilize_z=True, coord_ordering='yx'):
        '''
        Maybe in the future have a constructor also for 3D coordinates.
        '''
        
        self.coord = {} # Currently not populated
        self._c = self.coord # Just an alias
        
        if type(coord)==list:
            if coord[0].shape[1]==2:
                if coord_ordering=='yx':
                    self.coord['yx'] = coord
                    self.coord_2d = coord
                    self.nFrame = len(coord)    
                    self.nInFrame = np.array([c.shape[0] for c in coord])
                else:
                    "coord_ordering not yet implemented in the constructor"
        
        self.coord_3d_dz1_fl, self.nInVolume = \
                self._conv_coord_2d_to_3d(self.coord_2d, volFrame0, dz=1.0)
        
        self.volFrame0 = volFrame0
        self.zOfFrame = zOfFrame
        self.curvature = properties['curvature']
        self.curvatureVSplit = self._unflatten(self.curvature, self.nInVolume)
        self.boxIndices = properties['boxIndices']
        self.boxNPlane = properties['boxNPlane']
        
        if stabilize_z:
            self.coord_3d_dz1_fl = self._stabilize_z(self.coord_3d_dz1_fl, 
                self.curvatureVSplit,
                nPlane=self.boxNPlane, boxIndices=self.boxIndices,
                method="xyMaxCurvature")
                
            self.coord_2d, self.nInFrame = self._conv_coord_3d_to_2d(
                                           self.coord_3d_dz1_fl, self.volFrame0)
        
        self.coord_3d, self.nInVolume = self._conv_coord_2d_to_3d(self.coord_2d, 
                            volFrame0, zOfFrame=self.zOfFrame)
        self.coord_3d_xyz, self.nInVolume = self._conv_coord_2d_to_3d(
                            self.coord_2d, volFrame0, zOfFrame=self.zOfFrame,
                            coord_3d_ordering='xyz')
        self.coord_3d_dz1, self.nInVolume = \
                self._conv_coord_2d_to_3d(self.coord_2d, volFrame0, dz=1, dtype=int)
    
    def __call__(self):
        print("hi. use the __call__ method to have the irregular strides.")    
            
    '''
    def update_coord_from_XY(self,
    
    def update_coord_from_XYZ(self, '''
    
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
        coord_3d: list of numpy arrays
            coord_3d[l][j, n] gives the coordinate n of neuron j in volume l.
        nInVolume: numpy array
            nInVolume[l] gives the number of neurons in volume l.
        '''

        # Make sure nInFrame is an integer and can be used as an index.
        #nInFrame = nInFrame.astype(int, copy=False)
        nInFrame = np.array([c.shape[0] for c in coord_2d])
        
        # Initialize the lists containing the neurons coordinates.
        coord_3d = []
        nInVolume = []
        L = len(volFrame0)-1
                
        #For each volume
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
                 
            coord_3d.append(NeuronInVolume)
            
        return coord_3d, nInVolume
    
    
    @staticmethod    
    def _conv_coord_3d_to_2d(coord_3d, volFrame0, coord_3d_ordering='zyx', 
        coord_2d_ordering='yx'):
        '''
        Converts coordinates from a list of np.array([[z,y,x],]) for each 
        volume to a list of np.array([[y,x],]) for one frame, with the 
        corresponding number of neurons in each frame.
        
        Parameters
        ----------
        coord_3d: list of numpy arrays
            coord_3d[l][j, n] gives the coordinate n of neuron j in volume l.
            The z coordinate must be the index of the corresponding frame, and
            not the actual coordinate.
        volFrame0: numpy array
            volFrame0[l] gives the first frame of volume l. As last element, it
            must contain also the last+1 frame of the last volume.
        coord_3d_ordering: string
            Specifies the ordering of the coordinates inside the input array.
            Default is 'zyx', (indexing ordering).
        coord_2d_ordering: string
            Specifies the ordering of the coordinates inside the input array.
            Default is 'yx' (indexing order). For plotting use 'xy'.
            
        Returns
        -------
        coord_2d: list of numpy arrays
            coord_2d[i][j,n] gives the coordinate n of neuron j in frame i
        nInFrame: numpy array
            nInFrame[i] gives the number of neurons in frame i
        
        '''
        nInFrame = []
        coord_2d = []
        nVolume = len(coord_3d)
        
        for l in np.arange(nVolume):
            Brain = np.rint(coord_3d[l]).astype(np.int)
            for f in np.arange(volFrame0[l+1]-volFrame0[l]):
                
                # Depending on the specified ordering, extract the x, y, and z 
                # from coord_3d. z is used to select which points to assign to
                # this frame.
                # Out of here, you get the 2d coord with order xy.
                if coord_3d_ordering=='xyz':
                    ii = np.where(Brain[:,2]==f)
                    tmp = Brain[ii,0:2][0]
                elif coord_3d_ordering=='zyx':
                    ii = np.where(Brain[:,0]==f)
                    tmp = Brain[ii,1:3][0][:,::-1]
                
                # Flip x and y if the requested ordering requires it.
                if coord_2d_ordering=='yx':
                    tmp = tmp[:,::-1]
                
                coord_2d.append(np.array(tmp))
                nInFrame.append(len(ii[0]))

        return coord_2d, np.array(nInFrame)
    
    @staticmethod
    def _conv_coord_2d_to_1d(coord_2d, coord_2d_ordering='yx', size_x_f=256, 
        resize_i_to_f=2):
        '''
        Converts 2D coordinates to their 1D representation.
        
        Parameters
        ----------
        coord_2d: numpy array
            coord_2d[J,n] gives the coordinate n of neuron J
        coord_2d_ordering: string
            Specifies the ordering of the coordinates inside the input array.
            Default is 'yx', (indexing ordering).
        size_x_f: integer scalar
            Size along x of the intended 2D array. Can be of different scale 
            with respect to the original reference frame.
        resize_i_to_f: integer scalar
            Scale by which to divide the input coordinates to obtain the
            output coordinates (in 2D representation) in the intended 
            reference frame.
        
        Returns
        -------
        coord_1d: numpy array
            coord_1d[J] gives the 1D representation of the input coord_2d[J] in
            the intended reference frame. 
        '''
        
        if type(coord_2d)==list:
            coord_2d = _flatten(coord_2d)
        
        if not np.issubdtype(type(coord_2d[0,0]),np.integer):
            coord_2d = np.rint(coord_2d).astype(int)
        
        nNeurons = coord_2d.shape[0]
        coord_1d = np.empty(nNeurons,dtype=int)
        
        if coord_2d_ordering=='xy':
            Y = coord_2d[1]
            X = coord_2d[0]
        else:
            Y = coord_2d[0]
            X = coord_2d[1]
        
        coord_1d = Y//resize_i_to_f * size_x_f + X//resize_i_to_f
        
        return coord_1d
    
    @staticmethod    
    def _conv_coord_1d_to_2d(coord_1d, size_x_i=256, resize_f_to_i=2,
        coord_2d_ordering='yx',):
        '''
        Converts coordinates from their 1D representation to their 2D one.
        
        Parameters
        ----------
        coord_1d: numpy array
            Coordinates in their 1D representation.
        size_x_i: integer scalar
            Size along x of the original array/image.
        resize_f_to_i: integer scalar
            Scale by which to multiply the 2D representation of the input 1D
            coordinates to obtain the output coordinates (in 2D representation) 
            in the final reference frame.
        coord_2d_ordering: string
            Specifies the ordering of the coordinates inside the output array.
            Default is 'yx', (indexing ordering).
            
        Returns
        -------
        coord_2d: numpy array
            coord_2d[J,n] gives the coordinate n of neuron J.
        '''
        
        Y = coord_1d//size_x_i
        X = coord_1d - Y*size_x_i
        
        if coord_2d_ordering=='xy':
            coord_2d = np.copy(np.array([X,Y]).T)
        else:
            coord_2d = np.copy(np.array([Y,X]).T)
            
        return coord_2d
        
        
    @staticmethod
    def _flatten(coord):
        '''
        Takes a list of numpy arrays of coordinates, each list corresponding to
        a block (e.g., a frame or a volume), and returns a numpy array
        containing all the coordinates together with the number of coordinates
        contained in each of the original blocks.
        The output coordinates are in the same order as the input ones, so the 
        original structure can be retrieved.
        
        Parameters
        ----------
        coord: list of numpy arrays
            coord[l][j, n] gives the coordinate n of neuron j in block l. The
            input coordinates can be 2D or 3D.
        
        Returns
        -------
        coord_fl: numpy array
            coord_fl[J, n] gives the coordinate n of neuron J, with J the index
            of the neuron counting from neuron 0 of block 0 in the input
            coordinates.
        nInBlock: numpy array
            nInBlock[l] gives the number of neurons in block l in the original
            structure of coor.
            
        Note
        ----
        The global index J for neuron j in block l is given by
        nInPrevBlocks = np.zeros(len(nInBlock) + 1)
        nInPrevBlocks[1:] = np.cumsum(nInBlock)
        J = nInPrevBlocks[l]+j
        '''
        
        # Initialize the output array with the same number of coordinates and
        # data type as the input array
        # Find the first non-empty block
        for m in np.arange(len(coord)):
            if len(coord[m])!=0:
                break
        nCoord = coord[m].shape[1]
        coordType = type(coord[m][0,0])
        
        nInBlock = [len(coord_in_block) for coord_in_block in coord]
        nInBlock = np.array(nInBlock).astype(int)
        nNeurons = np.sum(nInBlock)
        
        coord_fl = np.empty((nNeurons,nCoord),dtype=coordType)
        
        nBlock = len(nInBlock)
        q = 0
        for l in np.arange(nBlock):
            coord_fl[q:q+nInBlock[l]] = coord[l]
            q += nInBlock[l]
            
        return coord_fl, nInBlock
        
    @staticmethod
    def _unflatten(coord, nInBlock):
        '''
        Takes a numpy array of coordinates, and unflattens them into a list of 
        numpy array, each containing the number of neurons specified by the
        elements of nInBlock. The global order of the neurons is preserved.
        
        Parameters
        ----------
        coord: numpy array
            coord[J,n] gives the coordinate n of neuron J, where J is the global
            index of the neuron. The input coordinates can be 2D or 3D.
        nInBlock: numpy array
            nInBlock[l] gives the number of neurons in block l.
            
        Returns
        -------
        coord_unfl: list of numpy arrays
            coord[l][j, n] gives the coordinate n of neuron j in block l.
        '''
        
        coord_unfl = []
        q = 0
        for ninblock in nInBlock:
            coord_unfl.append(coord[q:q+ninblock])
            q += ninblock
        
        return coord_unfl
        
    
    @staticmethod
    def _stabilize_z(coord_3d, curvatureVSplit, nPlane=7, boxIndices=
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
        coord_3d: list of numpy arrays
            coord_3d[l][j, n] gives the coordinate n of neuron j in volume l.
            The z coordinate must be the index of the corresponding frame, and
            not the actual coordinate.
        curvatureVSplit: list of numpy arrays
            curvatureVSplit[l][j, m] gives the curvature at point m in the box
            (described by nPlane and boxIndices) around neuron j in volume l.
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
        coord_3d_out: list of numpy arrays
            coord_3d_out[l][j, n] gives the coordinate n of the stabilized neuron j 
            in volume l.
        '''
    
        # Initialize output list
        coord_3d_out = []
        
        # Determine the index of the z coordinate in the input and output arrays
        z_indices = {"zyx":0,"xyz":2}
        z_index = z_indices[coord_3d_ordering]
        
        # Build z range around 0 for specified nPlanes
        z = np.arange(-(nPlane//2),nPlane//2+1,dtype=np.float)
        # For each volume
        for l in np.arange(len(coord_3d)):
            Brain = coord_3d[l]
            Curvature = curvatureVSplit[l]
            Curv = np.zeros((Curvature.shape[0],nPlane))
            
            # Depending on the method, build an array Curv of the curvature to be
            # used for each plane.
            if method=="xyMaxCurvature":
                for pl in np.arange(nPlane):
                    c = np.max(Curvature[:,boxIndices[pl]],axis=1)
                    Curv[:,pl] = c
            else:
                centralIndices = np.zeros(nPlane)
                for pl in np.arange(nPlanes):
                    sh0 = boxIndices[pl].shape[0]
                    centralIndices[pl] = boxIndices[pl][sh0//2]
                Curv = Curvature[:,CentralIndices] #look just along z
            
            Brain[:,z_index] += np.sum(z*Curv,axis=1)/np.sum(Curv,axis=1)
            coord_3d_out.append(Brain)

        return coord_3d_out
