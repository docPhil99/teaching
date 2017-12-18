# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:27:14 2017

@author: phil
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("demoEdges.pyx")
)