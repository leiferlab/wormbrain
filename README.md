# wormbrain
This package contains code to handle small point sets representing small brains. It provides a class, Brains, to represent coordinates of a sequence of instances of the same brain, for example the same brain at different time points; functions to geometrically register pointsets; and functions to match points in two point sets.

## Installation
To install the module, use (with `python` an alias for python3)
```
python -m pip install . --user
```
adding `--user` at the end if you want to install it in your home folder, or do not have privileges to write files in the standard target folder. Note that there is a C++ extension: the setup.py script will take care of its compilation, but you need to have a C++ compiler installed on your system.

### Dependencies
The modules in the package depend on standard modules like os, pkg_resources, re, copy, numpy, matplotlib, mpl_toolkits, and json, and the less standard module mistofrutta (from francescorandi/mistofrutta) for the irregular array irrarray structure and some plotting.

The package contains a C++ extension that depends on Eigen/Dense and boost/math/special_functions/digamma. For the Leifer Lab, once you set up the correct PATH variable (see wiki), these libraries are already in the shared disk space.

