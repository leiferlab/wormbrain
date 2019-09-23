#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

_dsmm_c = Extension('wormbrain.reg._dsmm_c',
                    sources = ['dsmm/dsmm.cpp','dsmm/dsmm_utils.cpp','wormbrain/reg/_dsmm_c.cpp'],
                    include_dirs = [
                        'dsmm',
                        numpy.get_include()
                        ],
                    extra_compile_args=['-ffast-math','-O3'])#


setup(name='wormbrain',
      version='0.1',
      description='Handling small brains',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['wormbrain','wormbrain.reg'],
      ext_modules = [_dsmm_c]
     )
