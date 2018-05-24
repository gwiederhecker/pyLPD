# -*- coding: utf-8 -*-
"""
**pyLPD**

This package compiles several functions used at our
laboratory (http://sites.ifi.unicamp.br/lpd).

These goal is to build three major submodules:

    * MLtools : interaction with matlab, matlab-like functions
        Basic functions adapted from Matlab to allow typical data-processing
    * VISAtools : instrument control (not ready yet)
        Toolset to control many of the available instrumentation.
        This is based heavily on PyVISA.
    * Simtools : simulation-support scripts (not ready yet)
        Basic functions for simple simulations and COMSOL post-processing

Examples
--------
    >>> from pyLPD import MLtools as mlt   # import the MLtools module as mlt
    >>> from pyLPD import VISAtools as vlt   # import the MLtools module as mlt
    >>> from pyLPD import SIMtools as smt   # import the MLtools module as mlt

"""


__version__ = '0.0.1.6'  


## import instrument modules
from .MLtools import *
#import .MLtools
