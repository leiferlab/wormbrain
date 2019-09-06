#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

_dsmm_c = Extension('wormbrain.reg._dsmm_c',
                    sources = ['dsmm/dsmm.cpp','dsmm/dsmm_utils.cpp','wormbrain/reg/_dsmm_c.cpp'],
                    include_dirs = [
                        'dsmm',
                        '/home/francesco/.local/lib/boost_1_71_0',
                        '/home/francesco/.local/lib/eigen',
						'../eigen',
						'../boost',
						'/projects/LEIFER/lib/include/eigen',
						'/projects/LEIFER/lib/include/boost',
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
