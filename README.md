# wormbrain
This package contains code to handle small point sets representing small brains. It provides a class, Brains, to represent coordinates of a sequence of instances of the same brain, for example the same brain at different time points; functions to to match points in two point sets; and functions geometrically register pointsets.

## Installation
To install the module, use (with `python` an alias for python3)
```
python -m pip install . --user
```
adding `--user` at the end if you want to install it in your home folder, or do not have privileges to write files in the standard target folder. Note that there is a C++ extension: the setup.py script will take care of its compilation, but you need to have a C++ compiler installed on your system.

### Dependencies
The modules in the package depend on standard modules like os, pkg_resources, re, copy, numpy, matplotlib, mpl_toolkits, and json, and the less standard module mistofrutta (from francescorandi/mistofrutta) for the irregular array irrarray structure and some plotting.

The package contains a C++ extension that depends on Eigen/Dense and boost/math/special_functions/digamma. For the Leifer Lab, once you set up the correct PATH variable (see wiki, Della and Tigress, .bashrc), these libraries are already in the shared disk space.

## Usage
### wormbrain Python package
For more detailed documentation, see the docs at docs/wormbrain/index.html.
- The Brains class is used to represent the coordinates of multiple instances of the same brain, like you have in a whole-brain imaging recording. Because the number of detected neurons in each volume can fluctuate, the class uses the irregular array (irrarray) from the mistofrutta package (github.com/francescorandi/mistofrutta). The class also has methods for coordinate conversions, and stores information about how the neuron coordinates have been extracted and preprocessed. After instantiation, the main methods used will be the direct call of the object, trueCoords(), and plot().
- The match module contains functions to match neurons belonging to two different instances of the same brain. The matching can be performed using different criteria, and after different registrations between the pointsets (see the reg module). The main function is match(), which is an interface to both different matching *and* registration methods. Other functions are for saving and loading matching results to and from file, and for plotting.
- The reg module contains the functions for the different registration methods. In addition to simple linear registrations, it contains two implementations of the Dirichelet-Student's t Mixture Model (DSMM) originally published by Zhou et al., Scientific Reports 2018. One of the implementations is fully written in Python, the other one is in C++ and is wrapped in Python (see dsmm).
### dsmm C++ implementation
The repository also contains the C++ implementation of the DSMM registration method (Zhou et al., implementation by Francesco Randi with some corrections). The C++ code is compiled automatically by pip and the setup script. For details about the algorithm and the implementation, see the docs of the C++ function.

## Author
Francesco Randi @ Leifer Lab, Princeton Physics Department
