#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import git

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
    "scipy",
]

# Get git commit info to build version number/tag
repo = git.Repo('.git')
git_hash = repo.head.object.hexsha
git_url = repo.remotes.origin.url
v = repo.git.describe(always=True)
if repo.is_dirty(): v += ".dirty"

setup(
    name="wormbrain",
    version=v,
    description="Handling small brains",
    author="Francesco Randi",
    author_email="francesco.randi@gmail.com",
    packages=["wormbrain", "wormbrain.reg"],
    cmdclass={"build_ext": CustomBuildExtCommand},
    install_requires=requirements,
    extras_require={"dev": ["pytest", "pytest-cov", "hypothesis"]},
    dependency_links=['git+https://github.com/francescorandi/mistofrutta'],
    ext_modules=ext_modules,
    package_data={'wormbrain': ['CELL_ID_visualize_light_class_def/+Image/*.m',
                                'CELL_ID_visualize_light_class_def/+Neurons/*.m',
                                'from_visualize_light.m',
                                'to_visualize_light.m',
                  ]},
)
