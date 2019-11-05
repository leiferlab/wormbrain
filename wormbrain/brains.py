import numpy as np
import matplotlib.pyplot as plt
import mistofrutta.struct.irrarray as irrarray
from copy import deepcopy as deepcopy
import json
import re

class Brains:
    '''Container for neuron coordinates. It depends on the arrays with 
    "irregular" strides from the repository 
    github.org/francescorandi/mistofrutta, which are just a shorter notation 
    for specific types of slices of numpy arrays.
    
    Creation methods, besides the __init__, inlude from_find_neurons() and 
    from_file(). Can be saved to a json file with to_file().
    
    The coordinates of the neurons can be accessed in various ways: With 
    Cervelli an instance of the class
    >>> Cervelli[i]
    will return the i-th point (counting from the very beginning in the whole
    set of points passed). The coordinates are stored in indexing-order so,
    for 3D points, this will return z,y,x. z is the frame index inside the
    volume. For a rescaled version of z, see trueCoords().
    Upon call, the internal irrarray object which 
    stores the coordinates is called, giving direct access to all its
    functionalities. To obtain the coordinatesof neurons contained in volume m, 
    use
    >>> Cervelli(vol=m)     
    (see mistofrutta.struct.irrarray for further documentation).
    >>> Cervelli.trueCoords(m)
    will return the "true" coordinates, in which z has the same dimensions as
    y and x.
    
    Upon initialization, if the appropriate dictionary is passed via the
    properties parameter, the z-coordinates are "stabilized" using the
    information about the local curvature around that neuron.
    '''
    
    filename = "brains.json"
    
    def __init__(self, coordZYX, nInVolume, zOfFrame=None, properties={}, 
                 stabilize_z=True):

        #coordZYX, self.nInVolume, self.nInFrame = self._conv_coord_2d_to_3d(coord, 
        #                                volFrame0, dtype=int)
        self.nInVolume = nInVolume
        self.coord = irrarray(coordZYX, [self.nInVolume], strideNames=["vol"])
        #self.volFrame0 = volFrame0
        
        if zOfFrame == None: zOfFrame = np.zeros((len(self.nInVolume),1))
        self.zOfFrame = zOfFrame
        
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
                    method="xyAvgCurvature")
                self.coord = self._stabilize_x(self.coord, self.curvature, nPixelsMax=5)
                self.coord = self._stabilize_y(self.coord, self.curvature)
                    
                self.coord = np.rint(self.coord)
            
        self.coord = self.coord.astype(int)
    
    @classmethod
    def from_find_neurons(cls, coord, volFrame0=None, *args, **kwargs):
        '''Create a Brains object from the result of the results of 
        wormneuronsegmentation.findNeurons().
        
        Parameters
        ----------
        coord: list of numpy array
            coord[i][j,coord] gives the coordinate (y,x) of point j in frame i.
        volFrame0: numpy array, optional
            volFrame0[m] gives the first frame of volume m. If passing results
            for M neurons, volFrame0 has to contain also the first frame of
            volume +1 (or the 1+last frame of volume M).
            If passing the results for a sequence of single frames (and not a 
            volumetric recording) set to None (or don't pass anything).
        
        Returns
        -------
        Instance of class.
        '''
        if volFrame0 is None:
            volFrame0 = np.arange(len(coord)+1,dtype=int)
        try:
            rectype = kwargs.pop('rectype')
        except:
            rectype = "3d"
        if rectype == "2d":
            kwargs['stabilize_z'] = False

        coordZYX, nInVolume, nInFrame = cls._conv_coord_2d_to_3d(coord, 
                                        volFrame0, dtype=int)
                                        
        return cls(coordZYX, nInVolume, *args, **kwargs)
    
    @classmethod
    def from_file(cls, folder, filename=""):
        '''Create a Brains object loading the data from a previously created
        json file.
        
        Parameters
        ----------
        folder: string
            Folder containing the file.
        
        Returns
        -------
        Instance of class.
        '''
        # for future multiple methods from loading from different formats
        #ext = filename.split(".")[-1]
        
        if folder[-1]!="/": folder += "/"
        
        if filename=="":
            filename = cls.filename
        f = open(folder+filename)
        c = json.load(f)
        f.close()
        
        coordZYX = np.array(c['coordZYX'])
        nInVolume = np.array(c['nInVolume'])
        zOfFrame = [np.array(z) for z in c['zOfFrame']]
        properties = {}
        try:
            props = c['properties']
            properties['curvature'] = [np.array(curv) for curv in props['curvature']]
            properties['boxIndices'] = [np.array(bi) for bi in props['boxIndices']]
            properties['boxNPlane'] = props['boxNPlane']
        except:
            pass
        
        # Don't do any implicit stabilization if loaded from file. 
        stabilize_z = False 
        
        return cls(coordZYX, nInVolume, zOfFrame, properties, stabilize_z)
    
    def append(self, brains2):
        '''Append to this object the content of another instance of Brains.
        It does not perform any z-stabilization. Assumes the curvature in 
        brains2 was extracted the same way as in this instance.
        (Modify stuff mimicking __init__()).
        '''
        self.nInVolume = np.append(self.nInVolume, brains2.nInVolume)
        self.coord = np.append(self.coord, brains2.coord, axis=0)
        self.coord = irrarray(self.coord, [self.nInVolume], strideNames=["vol"])
        
        # Concatenate
        self.zOfFrame = self.zOfFrame + brains2.zOfFrame
        
        self.curvature = np.append(self.curvature, brains2.curvature)
        self.curvature = irrarray(self.curvature, self.nInVolume, 
                                        strideNames=["vol"])
        
        self.coord = self.coord.astype(int)
        
    
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
    
    def to_file(self, foldername, filename=""):
        if foldername[-1]!="/": foldername += "/"
        
        diz = {}
        diz['coordZYX'] = [c.tolist() for c in self.coord]
        diz['nInVolume'] = self.nInVolume.tolist()
        diz['zOfFrame'] = [z.tolist() for z in self.zOfFrame]
        props = {}
        try:
            props['curvature'] = [c.tolist() for c in self.curvature]
            props['boxIndices'] = [c.tolist() for c in self.boxIndices]
            props['boxNPlane'] = self.boxNPlane
        except:
            pass
        diz['properties'] = props
        
        if filename=="":
            filename = self.filename
        
        output = json.dumps(diz, indent=4)
        # Prettify the json serialization 
        o1 = re.sub(r'\[\s+(\d)', r'[\1', output)
        o2 = re.sub(r'(\d),\s+(\d)', r'\1, \2', o1)
        o3 = re.sub(r'(\d)\s+\]',r'\1]',o2)
        
        f = open(foldername+filename,'w')
        f.write(o3)
        f.close()
        
    def trueCoords(self, vol, coord_ordering='zyx'):#, returnIrrarray=False):
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
                trueCoords[l] = np.copy(trueCoords[l][:,::-1],order="c")
                
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
        
        try:
            datatype=type(coord_2d[0][0])
        except:
            datatype=int
        
        coord_3d = np.zeros((nTotal,3),dtype=datatype)
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
            
        return coord_3d, np.array(nInVolume), np.array(nInFrame)
        
    def getOverlay(self, vol):
        try:
            bla = vol[0]
        except:
            vol = [vol]
        
        nVolume = len(vol)
        Overlay = []
        for kappa in np.arange(nVolume):
            cerv = self(vol=vol[kappa])
            for mu in np.arange(self.nInVolume[kappa]):
                Overlay.append(cerv[np.where(cerv[:,0]==mu)[0],1:][:,::-1])
                
        return Overlay
        
        
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
        elif method=="xyAvgCurvature":
            for pl in np.arange(nPlane):
                c = np.average(curvature[:,boxIndices[pl]],axis=1)
                curv[:,pl] = c
        else:
            centralIndices = np.zeros(nPlane)
            for pl in np.arange(nPlane):
                sh0 = boxIndices[pl].shape[0]
                centralIndices[pl] = boxIndices[pl][sh0//2]
            curv = curvature[:,centralIndices] #look just along z
        
        coord_3d_out = np.zeros_like(coord,dtype=np.float)
        if coord_3d_ordering=="zyx":
            coord_3d_out[:,1:3] = coord[:,1:3]
        else:
            coord_3d_out[:,0:2] = coord[:,0:2]
        coord_3d_out[:,z_index] = coord[:,z_index].astype(np.float) + np.sum(z*curv,axis=1)/np.sum(curv,axis=1)
        
        return coord_3d_out
    
    @staticmethod    
    def _stabilize_x(coord, curvature, nPixelsMax=5, boxIndices=
        [np.array([10,23,36]), np.array([2,7,11,15,29,24,28,33,37,41,46]), 
        np.array([0,1,3,5,6,8,12,16,18,19,21,25,29,31,32,34,38,42,44,45,47,49,50]),
        np.array([4,9,13,17,22,26,30,35,39,43,48]), np.array([14,27,40])], 
        coord_3d_ordering="zyx", method="curvatureAverage"):
        
        # Determine the index of the x coordinate in the input and output arrays
        x_indices = {"zyx":2,"xyz":0}
        x_index = x_indices[coord_3d_ordering]
        
        # Build z range around 0 for specified nPlanes
        x = np.arange(-(nPixelsMax//2),nPixelsMax//2+1,dtype=np.float)
        
        curv = np.zeros((coord.shape[0],nPixelsMax))
        if method=="curvatureAverage":
            for pl in np.arange(nPixelsMax):
                c = np.average(curvature[:,boxIndices[pl]],axis=1)
                curv[:,pl] = c
        
        coord_3d_out = np.zeros_like(coord,dtype=np.float)
        if coord_3d_ordering=="zyx":
            coord_3d_out[:,0:2] = coord[:,0:2]
        else:
            coord_3d_out[:,1:3] = coord[:,1:3]
        coord_3d_out[:,x_index] = coord[:,x_index].astype(np.float) + np.sum(x*curv,axis=1)/np.sum(curv,axis=1)
        
        return coord_3d_out
    
    @staticmethod    
    def _stabilize_y(coord, curvature, nPixelsMax=5, boxIndices=
        [np.array([6,19,32]), np.array([1,7,8,9,29,21,22,33,34,35,45]),
        np.array([0,2,3,4,10,11,12,13,14,23,24,25,26,27,36,37,38,39,40,46,47,48,50]),
        np.array([5,15,16,17,28,29,30,41,42,43,49]),np.array([18,31,44])],
        coord_3d_ordering="zyx", method="curvatureAverage"):
        
        y_index = 1
        
        # Build z range around 0 for specified nPlanes
        y = np.arange(-(nPixelsMax//2),nPixelsMax//2+1,dtype=np.float)
        
        curv = np.zeros((coord.shape[0],nPixelsMax))
        if method=="curvatureAverage":
            for pl in np.arange(nPixelsMax):
                c = np.average(curvature[:,boxIndices[pl]],axis=1)
                curv[:,pl] = c
        
        coord_3d_out = np.zeros_like(coord,dtype=np.float)
        coord_3d_out[:,0] = coord[:,0]
        coord_3d_out[:,2] = coord[:,2]
        coord_3d_out[:,y_index] = coord[:,y_index].astype(np.float) + np.sum(y*curv,axis=1)/np.sum(curv,axis=1)
        
        return coord_3d_out
        
    def fit_sphere(self):
        curvature = self.curvature
        boxIndices = self.boxIndices
        boxNPlane = self.boxNPlane
        
        # Calculate weights(z) [i.e. curvature] taking the maximum curvature 
        # in each plane.
        curv = np.zeros((self.coord.shape[0],boxNPlane))
        for pl in np.arange(boxNPlane):
            c = np.max(curvature[:,boxIndices[pl]],axis=1)
            curv[:,pl] = c
        
        # Use the value in plane nPlane//2+1 and nPlane//2+2 to "fit" the radius
        # of the sphere for each neuron
        R = self._sphere_radius(curv[:,boxNPlane//2+1:boxNPlane//2+2+1]/curv[:,boxNPlane//2+1,None])
        
        # Calculate the ratio between the curvature in the central plane and
        # the radius. This is the number by which you need to multiply the
        # signal to get the peak signal out.
        
        # This has become just the radius because I fitted the values after
        # normalizing them
        #yOverR = curv[:,boxNPlane//2+1]/R
        yOverR = 1./R
        
        return irrarray(yOverR, self.nInVolume, strideNames=["vol"])
        
    @staticmethod
    def _sphere_radius(y, dx=1.0):
        y0sq = y[:,0]**2
        return np.sqrt(y0sq + ((y0sq-y[:,1]**2-dx**2)/(2.0*dx))**2)
        
    def plot(self, indices,mode='3d',plotNow=True,**kwargs):
        try:
            len(indices)
        except:
            indices = np.array([indices])
        
        if mode=='3d':
            fig, ax = self._plot_3d(indices, **kwargs)
        if mode=='2d':
            fig, ax = self._plot_2d(indices, **kwargs)
            
        if plotNow==True:
            plt.show()
            return
        else:
            return fig, ax
            
            
    def _plot_3d(self, indices, **kwargs):
        cfn = plt.gcf().number
        if len(plt.gcf().axes)!=0: cfn += 1
        
        showAll=True
        if 'showAll' in kwargs: showAll=kwargs['showAll']
        
        fig = plt.figure(cfn)
        ax = fig.add_subplot(111,projection='3d')
        
        for index in indices:
            brain = self.trueCoords(index)
            ax.scatter(brain.T[2],brain.T[1],brain.T[0],'o')
                
        return fig, ax
        

    def _plot_2d(self, indices, **kwargs):
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
