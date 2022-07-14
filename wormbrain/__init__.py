'''Components of the package:

- The Brains class is used to represent the coordinates of multiple instances of
  the same brain, like you have in a whole-brain imaging recording. Because the 
  number of detected neurons in each volume can fluctuate, the class uses the 
  irregular array (irrarray) from the mistofrutta package 
  (github.com/francescorandi/mistofrutta). The class also has methods for 
  coordinate conversions, and stores information about how the neuron 
  coordinates have been extracted and preprocessed. After instantiation, the 
  main methods used will be the direct call of the object, trueCoords(), and 
  plot().
  
- The match module contains functions to match neurons belonging to two 
  different instances of the same brain. The matching can be performed using 
  different criteria, and after different registrations between the pointsets 
  (see the reg module). The main function is match(), which is an interface to 
  both different matching *and* registration methods. Other functions are for 
  saving and loading matching results to and from file, and for plotting.
  
- The reg module contains the functions for the different registration methods. 
  In addition to simple linear registrations, it contains two implementations of 
  the Dirichelet-Student's t Mixture Model (DSMM) originally published by Zhou et 
  al., Scientific Reports 2018. One of the implementations is fully written in 
  Python, the other one is in C++ and is wrapped in Python (see dsmm).
  
Author: Francesco Randi @ Leifer Lab, Princeton Physics Department
'''


__all__ = ['Brains','match','reg']
__pdoc__ = {}
__pdoc__['wormbrain.Brains._plot_3d'] = True
__pdoc__['wormbrain.Brains._plot_2d'] = True

from .brains import Brains
from . import match
from . import reg
