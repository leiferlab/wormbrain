#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


ext_modules = [
    Extension(
        "wormbrain.reg._dsmm_c",
        sources=["dsmm/dsmm.cpp", "dsmm/dsmm_utils.cpp", "wormbrain/reg/_dsmm_c.cpp"],
        include_dirs=["dsmm"],
        extra_compile_args=["-ffast-math", "-O3"],
    )
]

requirements = [
    "numpy",
    "matplotlib",
    "pyqt5",
    "scipy",
]


setup(
    name="wormbrain",
    version="0.1",
    description="Handling small brains",
    author="Francesco Randi",
    author_email="francesco.randi@gmail.com",
    packages=["wormbrain", "wormbrain.reg"],
    cmdclass={"build_ext": CustomBuildExtCommand},
    install_requires=requirements,
    extras_require={"dev": ["pytest", "pytest-cov"]},
    dependency_links=['git+https://github.com/francescorandi/mistofrutta'],
    ext_modules=ext_modules,
)
