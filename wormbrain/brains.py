'''The class Brains is made available in the wormbrain namespace.'''

import numpy as np
import matplotlib.pyplot as plt
import mistofrutta.struct.irrarray as irrarray
from copy import deepcopy as deepcopy
import ujson as json
import re
import pkg_resources
import wormbrain as wormb
import os
import pickle

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
    "Default destination file name for the json version of the object"
    coord_filename = "transformed_neurons.txt"
    
    nInVolume = 0
    "Number of neurons in each volume"
    coord = irrarray(np.zeros(1), [np.array([1])], strideNames=["vol"])
    "Irregular array containing the coordinates of the neurons"
    zOfFrame = np.zeros(1)
    "z coordinate of each frame in each volume"
    labels = []
    "Labels of the neurons"
    
    version = ""
    "Version of the wormbrain module in use when instantiating the class."
    info = {}
    "Other info about the object"
    
    curvature = irrarray(np.zeros(1), [np.array([1])], strideNames=["vol"])
    "Irregular array containing the local curvature at the neural positions.\
    Note: x1/2 coordinate scale."
    boxIndices = np.zeros(1)
    boxIndicesX = np.zeros(1)
    boxIndicesY = np.zeros(1)
    boxNPlane = 0
    
    segmParam = {}
    "Parameters used for the segmentation of the neurons stored in the object."
    info = {}
    
    MMatch = None
    
    coord_upToIndex = {}
    coord_columnNames = []
    
    
    def __init__(self, coordZYX, nInVolume, zOfFrame=None, properties={}, 
                 stabilize_z=True,stabilize_xy=True):
                 #save_to_cache=False,save_to_path=None):
        '''The constructor transforms the coordinates array into an irrarray
        and, if requested to, "stabilizes" the coordinates based on the 
        local curvature around the neurons.
        
        Parameters
        ----------
        coordZYX: numpy array
            Array containing the coordinates of the neurons, in zyx order.
            coordZYX[i] are the coordinates of the i-th neuron in the whole
            serialized sequence of brains.
        nInVolume: numpy array
            nInVolume[j] is the number of neurons in neuron j. Used to split 
            the neurons into their respective volumes.
        zOfFrame: list of numpy arrays (optional)
            zOfFrame[j][l] is the "real" z coordinate of frame l in volume j.
            It is most useful if z has the same scale as x and y. Not required
            for 2D recordings in which each volume is composed of a single
            frame. Default: None.
        properties: dictionary (optional)
            Dictionary that can have the keys 'curvature', 'boxIndices', 
            'boxIndicesX', 'boxIndicesY', 'boxNPlane', 'segmParam', 'version'.
            NeuronProperties from wormneuronsegmentation.findNeurons() can be
            passed as properties. Default: {}.
        stabilize_z: bool (optional)
            If True, the z position is stabilized based on the local curvature
            around the neurons. Default: True.
        stabilize_xy: bool (optional)
            If True, the xy position is stabilized based on the local curvature
            around the neurons. Default: True.
        '''
        
        #coordZYX, self.nInVolume, self.nInFrame = self._conv_coord_2d_to_3d(coord, 
        #                                volFrame0, dtype=int)
        self.nInVolume = nInVolume
        self.coord = irrarray(coordZYX, [self.nInVolume], strideNames=["vol"])
        #self.volFrame0 = volFrame0
        
        if zOfFrame == None: zOfFrame = np.zeros((len(self.nInVolume),1))
        self.zOfFrame = zOfFrame
        
        self.version = pkg_resources.get_distribution("wormbrain").version
        self.info = {}
        self.info['version'] = self.version
        
        self.labels = [[] for jj in np.arange(len(nInVolume))]
        
        if len(properties.keys())!=0:
            self.boxIndices = properties['boxIndices']
            curvature_elements = 0
            for bi in self.boxIndices: curvature_elements += len(bi)
            self.curvature = np.array(properties['curvature'])
            
            if len(self.curvature.shape) == 1:
                self.curvature = self.curvature.reshape(
                                self.curvature.shape[0]//curvature_elements,
                                curvature_elements)
            
            self.curvature = irrarray(self.curvature, self.nInVolume, 
                                        strideNames=["vol"])
            
            try:
                self.boxIndicesX = properties['boxIndicesX']
                self.boxIndicesY = properties['boxIndicesY']
            except:
                pass
                
            self.boxNPlane = properties['boxNPlane']
            self.segmParam = properties['segmParam']
            self.info['segm_param'] = self.segmParam
            
            try:
                self.version = properties['version']
            except:
                pass
        
            if stabilize_z:
                self.coord = self._stabilize_z(self.coord, 
                    self.curvature,
                    nPlane=self.boxNPlane, boxIndices=self.boxIndices,
                    method="xyAvgCurvature")
            if stabilize_xy:
                self.coord = self._stabilize_x(self.coord, self.curvature, 
                                    nPixelsMax=5, boxIndices=self.boxIndicesX)
                self.coord = self._stabilize_y(self.coord, self.curvature,
                                    boxIndices=self.boxIndicesY)
                    
                self.coord = np.rint(self.coord)
            
        self.coord = self.coord.astype(int)
        
        #if save_to_cache and save_to_path is not None:
        #    f = open(save_to_path,"wb")
        #    pickle.dump(self,f)
        #    f.close()
        
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
        filename: string (optional)
            Name of the file containing the signal. Default: "", which is
            translated into the default filename for the class.
        
        Returns
        -------
        Instance of class.
        '''
        # for future multiple methods from loading from different formats
        #ext = filename.split(".")[-1]
        
        if folder[-1]!="/": folder += "/"
        
        if filename=="":
            filename = cls.filename
        
        # If pickled cache is available, load from there;
        # Otherwise parse the json file.
        pickle_filename = ".".join([filename.split(".")[0],"pickle"])
        cache_present = os.path.isfile(folder+pickle_filename)
        if False:#cache_present:
            f = open(folder+pickle_filename,"rb")
            obj = pickle.load(f)
            f.close()
            coordZYX = obj["coordZYX"]
            nInVolume = obj["nInVolume"]
            zOfFrame = obj["zOfFrame"]
            properties = obj["properties"]
        else:        
            f = open(folder+filename)
            c = json.load(f)
            f.close()
            
            coordZYX = np.array(c['coordZYX'])
            nInVolume = np.array(c['nInVolume'])
            zOfFrame = [np.array(z) for z in c['zOfFrame']]
            properties = {}
            
            # To be compatible with older versions of the file, there is this
            # sequence of try/except. At some point, this can be removed and all
            # the properties be loaded.
            try:
                props = c['properties']
                #properties['curvature'] = [np.array(curv) for curv in props['curvature']]
                properties['curvature'] = np.array(props['curvature'])
                properties['boxIndices'] = [np.array(bi) for bi in props['boxIndices']]
                properties['boxNPlane'] = props['boxNPlane']
            except:
                pass
                
            try:
                properties['boxIndicesX'] = [np.array(bi) for bi in props['boxIndicesX']]
                properties['boxIndicesY'] = [np.array(bi) for bi in props['boxIndicesY']]
            except:
                pass
                
            try:
                properties['segmParam'] = props['segmParam']
            except:
                pass
                
            try:
                properties['version'] = props['version']
            except:
                pass
                
            try:
                labels = c['labels']
            except:
                labels = None
                
            # cache in the pickled version
            #f = open(folder+pickle_filename,"wb")
            #pickle.dump({"coordZYX":coordZYX,"nInVolume":nInVolume,"zOfFrame":zOfFrame,"properties":properties},f)
            #f.close()
        
        # Don't do any implicit stabilization if loaded from file. 
        stabilize_z = False
        stabilize_xy = False 
        
        inst = cls(coordZYX, nInVolume, zOfFrame, properties, stabilize_z, stabilize_xy)
        
        if labels is not None:
            for j in np.arange(len(labels)):
                l = labels[j]
                inst.set_labels(j,l)
        
        return inst
        
    @classmethod
    def from_coord_file(cls, folder, filename=""):
        '''Create a Brains object from a file containing the coordinates of the
        neurons only.
        
        Parameters
        ----------
        folder: string
            Folder containing the file.
        filename: string (optional)
            Name of the file containing the coordinates of the neurons. 
            Default: "", which is translated to the default filename for the 
            class.
        
        '''
        if folder[-1]!="/": folder += "/"
        if filename == "": filename = cls.coord_filename
        
        f = open(folder+filename)
        s = f.readline()
        f.close()

        try:
            sp = s.split(";")
            n_volumes = int(sp[0].split("=")[1])
            n_neurons = int(sp[1].split("=")[1])
        except:
            s = json.loads(s[1:])
            n_volumes = s["n_volumes"]
            n_neurons = s["n_neurons"]
        nInVolume = np.ones(n_volumes)*n_neurons
        
        coordZYX = np.loadtxt(folder+filename)
        
        return cls(coordZYX, nInVolume, stabilize_z=False, stabilize_xy=False)
    
    def append(self, brains2):
        '''Append to this object the content of another instance of Brains.
        It does not perform any z-stabilization. Assumes the curvature in 
        brains2 was extracted the same way as in this instance.
        (Modify stuff mimicking __init__()).
        
        Parameters
        ----------
        brains2: Brains object
            Brains object to be appended to this.
        
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
        
    def add_points(self, points, vol=0):
        '''Append points to brain. Currently supported only for single brain
        object. The function does not supplement additional information in the
        properties, like the curvature.'''
        
        if len(points.shape)==1:
            points = np.array([points])
        
        self.nInVolume[vol] += points.shape[0]
        self.coord = np.append(self.coord, points, axis=0)
        
        for i in np.arange(points.shape[0]):
            self.labels[vol].append("")
        
        
    def delete_points(self, indices, vol=0):
        try: len(indices)
        except: indices = [indices]
        indices = np.array(indices)
        
        indices[indices<0] = self.nInVolume[vol]+indices[indices<0]
        
        self.coord = np.delete(self.coord,indices,axis=0)
        self.nInVolume[vol] -= indices.shape[0]
    
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
        '''Deepcopy of this object.'''
        return deepcopy(self)
    
    def to_file(self, foldername, filename=""):
        '''Save this instance of the object to file. A Brains object saved 
        this way can be recreated in memory with the class method from_file().
        
        Parameters
        ----------
        foldername: string
            Destination folder.
        filename: string (optional)
            Name of destination file. Default: "", which is translated to the 
            default filename for the class.
        
        '''    
        if foldername[-1]!="/": foldername += "/"
        
        diz = {}
        diz['coordZYX'] = [c.tolist() for c in self.coord]
        diz['nInVolume'] = self.nInVolume.tolist()
        diz['zOfFrame'] = [z.tolist() for z in self.zOfFrame]
        diz['labels'] = self.labels
        props = {}
        try:
            props['curvature'] = [c.tolist() for c in self.curvature]
            #props['curvature'] = self.curvature.tolist()
            props['boxIndices'] = [c.tolist() for c in self.boxIndices]
            props['boxIndicesX'] = [c.tolist() for c in self.boxIndicesX]
            props['boxIndicesY'] = [c.tolist() for c in self.boxIndicesY]
            props['boxNPlane'] = self.boxNPlane
            props['segmParam'] = self.segmParam
            props['version'] = self.version
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
        
    def load_matches(self,folder):
        if self.MMatch is None:
            self.MMatch, self.MMatch_parameters = wormb.match.load_matches(folder) 
        
    def trueCoords(self, vol, coord_ordering='zyx'):#, returnIrrarray=False):
        '''Returns the coordinates of the neurons contained in the specified
        volumes replacing z with its actual values, from zOfFrame.
        
        Parameters
        ----------
        vol: int or list of int
            Indices of the volume(s) requested.
        coord_ordering: string (optional)
            Ordering of the coordinates to be returned. Allowed values:
            'zyx' for indexing order, and 'xyz' for plotting order. 
            Default: 'zyx'.
            
        Returns
        -------
        trueCoords: numpy array
            Coordinates of the neurons.
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
        
    def get_closest_neuron(self,volume,coord,inverse_match=True,z_true=False,coord_ordering='zyx'):
        
        if coord_ordering == "xyz": coord = coord[::-1]
        neurons = self.coord(vol=volume)
        
        if z_true:
            # Convert passed true z coordinate to index
            z_index = np.argmin(np.absolute(coord[0]-self.zOfFrame[volume]))
            coord = np.array([z_index,coord[1],coord[2]])
        
        closest = np.argmin(np.sum(np.power(neurons-coord[None,:],2),axis=-1))    
            
        if inverse_match:
            closest_ref = np.where(self.MMatch[volume]==closest)[0]
            if closest_ref.shape[0] >0:
                closest_ref = closest_ref[0]
            else:
                closest_ref = -1
            closest = closest_ref
        
        return closest
        
    def set_labels(self, vol, labels):
        self.labels[vol] = labels
        
    def get_labels(self, vol):
        if len(self.labels[vol])<self.nInVolume[vol]:
            for i in np.arange(self.nInVolume[vol]-len(self.labels[vol])):
                self.labels[vol].append("")
        return self.labels[vol]
        
    @staticmethod
    def _conv_coord_2d_to_3d(coord_2d, volFrame0, zOfFrame=[], dz=1, 
            dtype=np.float, coord_2d_ordering='yx', coord_3d_ordering='zyx'):
        '''Converts coordinates from a list of np.array([[y,x],]) for one frame 
        to a list of np.array([[z,y,x],]) for each volume, with the 
        corresponding number of neurons in each volume.
        
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
        
    def getOverlay(self, vol, folder="", returnLabels=False):
        '''Returns the list of numpy arrays to be used as Overlay in 
        mistofrutta.plt.hyperstack, together with the OverlayLabels. The labels
        are not the indices of the neurons in their volume, but are the indices
        of the matched neurons in the reference brain/volume. If the labels are
        requested this function, therefore, assumes that a match file exists 
        that can be loaded via wormbrain.match.load_matches().
        
        Parameters
        ----------
        vol: int or list of int
            Indices of the volume(s) requested.
        folder: string (optional)
            Folder containing the match file to be loaded with 
            wormbrain match.load_matches(). Required if returnLabels is True.
        returnLabels: bool (optional)
            If True, the labels are returned. Default: False
            
        Returns
        -------
        Overlay: list of numpy arrays
            Overlay[frame_index][i] are the y,x coordinates of neuron i in
            a given frame.
        OverlayLabels: list of integers
            OverlayLabels[frame_index][i] is the index of the neuron
            in the reference brain corresponding to the neuron i in the 
            specified frame. Returned if returnLabels is True.
        
        '''
        try:
            bla = vol[0]
        except:
            vol = [vol]
        
        nVolume = len(vol)
        Overlay = []
        OverlayLabels = []     
        
        # It was like this, with the loading of the file before this for.
        #for kappa in np.arange(nVolume):
        #    cerv = self(vol=vol[kappa])
        #    numFrames = len(self.zOfFrame[vol[kappa]]);
        #    match = np.asarray(MMatch_inv[kappa])
        #    for mu in np.arange(numFrames):
        #        Overlay.append(cerv[np.where(cerv[:,0]==mu)[0],1:][:,::-1])
        #        OverlayLabels.append(match[cerv[:,0]==mu]);
        #if returnLabels:        
        #    return Overlay, OverlayLabels
        #else:
        #    return Overlay
        
        for kappa in np.arange(nVolume):
                cerv = self(vol=vol[kappa])
                numFrames = len(self.zOfFrame[vol[kappa]]);
                for mu in np.arange(numFrames):
                    Overlay.append(cerv[np.where(cerv[:,0]==mu)[0],1:][:,::-1])
        
        if returnLabels:
            MMatch, info = wormb.match.load_matches(folder)
        
            # create an "inverse" matching object where the index is current vol and the value is ref vol
            MMatch_inv = []; 
            for kappa in np.arange(nVolume):
                temp = [None]*self.nInVolume[vol[kappa]];
                for ref, cur in enumerate(MMatch[vol[kappa]],0):
                    if cur >= 0:
                        temp[int(cur)] = ref;
                MMatch_inv.append(temp);
                
            for kappa in np.arange(nVolume):
                cerv = self(vol=vol[kappa])
                numFrames = len(self.zOfFrame[vol[kappa]]);
                match = np.asarray(MMatch_inv[kappa])
                for mu in np.arange(numFrames):
                    OverlayLabels.append(match[cerv[:,0]==mu]);
            
            return Overlay, OverlayLabels            
        else:
            return Overlay
            
    def get_overlay2(self,vol,return_labels=False,label_size=None,copy=True):
        if copy:
            overlay_ = self.coord(vol=vol).copy()
        else:
            overlay_ = self.coord(vol=vol)
        
        overlay = irrarray(overlay_, irrStrides = [[self.nInVolume[vol],0]], strideNames=["ch"])
        if return_labels:
            if label_size is not None:
                labels_ = [str(i).zfill(label_size) for i in np.arange(self.nInVolume[vol])]
            else:
                labels_ = [str(i) for i in np.arange(self.nInVolume[vol])]
                
            if len(self.labels)>0:
                for ll in np.arange(len(self.labels[vol])):
                    if self.labels[vol][ll]!='': labels_[ll] = self.labels[vol][ll]
                              
            labels = irrarray(labels_, irrStrides = [[self.nInVolume[vol],0]], strideNames=["ch"])
            return overlay, labels
        else:
            return overlay
            
        
    @staticmethod
    def _stabilize_z(coord, curvature, nPlane=7, boxIndices=
        [np.arange(1),np.arange(1,6),np.arange(6,19),np.arange(19,32),
        np.arange(32,45),np.arange(45,50),np.arange(50,51)], 
        coord_3d_ordering="zyx", method=""):
        '''Stabilizes the z position of the neuron using the local curvature 
        around that point. This helps in obtaining less fluctuating results when
        the neurons were found in 2D with brute-force check along z, i.e. with 
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
                # The curvature has to be flipped in sign (i.e. peak = max of
                # flipped curvature) and the resulting negative values clipped
                # to 0, so that only the central region of the neuron matters.
                flipped_curv = -1.0*curvature[:,boxIndices[pl]]
                np.clip(flipped_curv,0,None,flipped_curv)
                c = np.max(flipped_curv,axis=1)
                curv[:,pl] = c
        elif method=="xyAvgCurvature":
            for pl in np.arange(nPlane):
                # Flip and clip curvature (see above).
                flipped_curv = -1.0*curvature[:,boxIndices[pl]]
                np.clip(flipped_curv,0,None,flipped_curv)
                c = np.average(flipped_curv,axis=1)
                curv[:,pl] = c
        else:
            centralIndices = np.zeros(nPlane)
            for pl in np.arange(nPlane):
                sh0 = boxIndices[pl].shape[0]
                centralIndices[pl] = boxIndices[pl][sh0//2]
            curv = curvature[:,centralIndices] #look just along z
            curv *= -1.0
            np.clip(curv,0,None,curv)
        
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
        coord_3d_ordering="zyx", method="curvatureAverage", curvatureImageResize=2.0):
        '''Stabilization of the x coordinate, equivalent to _stabilize_z.'''
        
        # Determine the index of the x coordinate in the input and output arrays
        x_indices = {"zyx":2,"xyz":0}
        x_index = x_indices[coord_3d_ordering]
        
        # Build z range around 0 for specified nPlanes
        x = np.arange(-(nPixelsMax//2),nPixelsMax//2+1,dtype=np.float)
        
        curv = np.zeros((coord.shape[0],nPixelsMax))
        if method=="curvatureAverage":
            for pl in np.arange(nPixelsMax):
                # The curvature has to be flipped in sign (i.e. peak = max of
                # flipped curvature) and the resulting negative values clipped
                # to 0, so that only the central region of the neuron matters.
                flipped_curv = -1.0*curvature[:,boxIndices[pl]]
                np.clip(flipped_curv,0,None,flipped_curv)
                c = np.average(flipped_curv,axis=1)
                curv[:,pl] = c
        
        coord_3d_out = np.zeros_like(coord,dtype=np.float)
        if coord_3d_ordering=="zyx":
            coord_3d_out[:,0:2] = coord[:,0:2]
        else:
            coord_3d_out[:,1:3] = coord[:,1:3]
        
        coord_3d_out[:,x_index] = coord[:,x_index].astype(np.float) + np.sum(x*curv,axis=1)/np.sum(curv,axis=1) * curvatureImageResize
        
        return coord_3d_out
    
    @staticmethod    
    def _stabilize_y(coord, curvature, nPixelsMax=5, boxIndices=
        [np.array([6,19,32]), np.array([1,7,8,9,29,21,22,33,34,35,45]),
        np.array([0,2,3,4,10,11,12,13,14,23,24,25,26,27,36,37,38,39,40,46,47,48,50]),
        np.array([5,15,16,17,28,29,30,41,42,43,49]),np.array([18,31,44])],
        coord_3d_ordering="zyx", method="curvatureAverage", curvatureImageResize=2.0):
        '''Stabilization of the y coordinate, equivalent to _stabilize_z.'''
        
        y_index = 1
        
        # Build z range around 0 for specified nPlanes
        y = np.arange(-(nPixelsMax//2),nPixelsMax//2+1,dtype=np.float)
        
        curv = np.zeros((coord.shape[0],nPixelsMax))
        if method=="curvatureAverage":
            for pl in np.arange(nPixelsMax):
                # The curvature has to be flipped in sign (i.e. peak = max of
                # flipped curvature) and the resulting negative values clipped
                # to 0, so that only the central region of the neuron matters.
                flipped_curv = -1.0*curvature[:,boxIndices[pl]]
                np.clip(flipped_curv,0,None,flipped_curv)
                c = np.average(flipped_curv,axis=1)
                curv[:,pl] = c
        
        coord_3d_out = np.zeros_like(coord,dtype=np.float)
        coord_3d_out[:,0] = coord[:,0]
        coord_3d_out[:,2] = coord[:,2]
        coord_3d_out[:,y_index] = coord[:,y_index].astype(np.float) + np.sum(y*curv,axis=1)/np.sum(curv,axis=1)
        
        return coord_3d_out
        
    def fit_sphere(self):
        '''Fit each neuron as a sphere to stabilize its z position, which is
        undersampled with respect to the x and y due to rougher scanning along
        z with respect to the pixels. Currently not used.
        
        Returns
        -------
        yOverR: irrarray
            Inverse of the radius of each neuron. (?)
        '''
        curvature = self.curvature
        boxIndices = self.boxIndices
        boxNPlane = self.boxNPlane
        
        # Calculate weights(z) [i.e. curvature] taking the maximum curvature 
        # in each plane.
        curv = np.zeros((self.coord.shape[0],boxNPlane))
        for pl in np.arange(boxNPlane):
            # The curvature has to be flipped in sign (i.e. peak = max of
            # flipped curvature) and the resulting negative values clipped
            # to 0, so that only the central region of the neuron matters.
            flipped_curv = -1.0*curvature[:,boxIndices[pl]]
            np.clip(flipped_curv,0,None,flipped_curv)
            c = np.max(flipped_curv,axis=1)
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
        '''Plots the neurons in the specified volumes. Based on the selected
        mode, the function redirects the call to _plot_3d and _plot_2d.
        
        Parameters
        ----------
        indices: list (or numpy array)
            Indices of the volumes to plot
        mode: string (optional)
            Plot mode: 3d or 2d. Default: '3d'.
        plotNow: bool
            If True, the function plots immediately. If False, the function
            returns matplotlib figures and axes to be plotted at a later point
            in the script. Default: True.
        **kwargs: other
            Other parameters to be passed to _plot_3d or _plot_2d.
            
        Returns
        -------
        fig: matplotlib figure 
            Returned only if plotNow is False.
        ax: matplotlib axis 
            Returned only if plotNow is False.
        
        '''
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
            
            
    def _plot_3d(self, indices, mask=None, mask_color='r'):
        '''Produces a 3D plot of the requested volumes. To be used via the 
        plot() method.
        
        Parameters
        ----------
        indices: integer or list of integers
            Indices of the requested volumes.
        mask: numpy array or list of numpy arrays (optional)
            Indices of neurons to highlight. When indices is a list, i.e. 
            plotting multiple volumes, mask should also be a list of indices
            for each volume. Default: None.
        mask_color: string or list of strings(optional)
            Color of the highlighted neurons, in matplotlib-understandable
            format. Default: "r".
            
        Returns
        -------
        fig: matplotlib figure
        ax: matplotlib axis
        
        '''
        cfn = plt.gcf().number
        if len(plt.gcf().axes)!=0: cfn += 1
        
        fig = plt.figure(cfn)
        ax = fig.add_subplot(111,projection='3d')
        
        if mask is not None:
            try: len(mask[0])
            except: mask = [mask]
        
        Q = len(indices)
        for q in np.arange(Q):
            index = indices[q]
            brain = self.trueCoords(index)
            ax.scatter(brain.T[2],brain.T[1],brain.T[0],'o')
            if mask is not None:
                ax.scatter(brain[mask[q]].T[2],brain[mask[q]].T[1],brain[mask[q]].T[0],'o',c=mask_color)
                
        return fig, ax
        

    def _plot_2d(self, indices, mask=None, mask_color='r'):
        '''Produces a 2D plot of the requested volumes. To be used via the 
        plot() method.
        
        Parameters
        ----------
        indices: list of integers
            Indices of the requested volumes.
        mask: numpy array or list of numpy arrays (optional)
            Indices of neurons to highlight. When indices is a list, i.e. 
            plotting multiple volumes, mask should also be a list of indices
            for each volume. Default: None.
        mask_color: string (optional)
            Color of the highlighted neurons, in matplotlib-understandable
            format. Default: "r".
            
        Returns
        -------
        fig: matplotlib figure
        ax: matplotlib axis
        
        '''
        # _plot_2d: plots each of the neurons at there location for the volumes specified by indices (list)
        # Cervelli: brain object that you want to plot (this will become self in the actual class)
        # indices: list object with the volumes that you want to plot
        cfn = plt.gcf().number
        if len(plt.gcf().axes)!=0: cfn += 1
               
        fig = plt.figure(cfn)
        ax = fig.add_subplot(111)
        
        if mask is not None:
            try: len(mask[0])
            except: mask = [mask]
        
        Q = len(indices)
        for q in np.arange(Q):
            index = indices[q]
            brain = self.trueCoords(index)
            ax.scatter(brain.T[2],brain.T[1],marker='o')
            if mask is not None:
                ax.scatter(brain[mask[q]].T[2],brain[mask[q]].T[1],marker='o',c=mask_color)
                
        return fig, ax
