# -*- coding: utf-8 -*-
"""
PyLPD v0.0.1.5

We compile in this package many functions used at our
laboratory (LPD - sites.ifi.unicamp.br/LPD).

These goal is to build three major submodules:

    * MLtools : interaction with matlab, matlab-like functions
    * VISAtools: instrument control (not ready yet)
    * Simtools: simulation scripts (not ready yet)


Documentation
-------------


Usage
-------------
    >>> from pyLPD import MLtools   # definição do gerador de funções

"""

## import instrument modules
from .MLtools import *
import .MLtools
