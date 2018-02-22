#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of decorator functions"""

import numpy as np
from astropy.io import fits

__all__ = ['_hpmap', 'update_docstring']


# Decorator and decorated functions
def update_docstring(fct, skip=0, head_docstring=None, foot_docstring=None):
    """Update the docstring of a decorated function by insterting a docstring at the beginning

    Parameters
    ----------
    fct : fct
        the original function from which we extract the documentation
    skip : int
        number of line to skip in the original documentation
    head_docstring : str
        additionnal documentation to be insterted at the top
    foot_docstring : str
        additionnal documentation to be inserted at the end

    Returns
    -------
    fct
        the decorator function with the updated docstring
    """

    # Extract first line of the original function
    docstring = str(fct.__doc__).split('\n')[0]

    if head_docstring:
        docstring += head_docstring

    docstring += '\n' + '\n'.join(str(fct.__doc__).split('\n')[skip:])

    if foot_docstring:
        docstring += foot_docstring

    return docstring


def _hpmap(hphdu_func):
    """Will decorate the function taking hp_hdu function to be able to use it with hp_map, hp_header instead of an :class:`astropy.io.fits.ImageHDU` object

    Parameters
    ----------
    a function using hp_hpu :class:`astropy.io.fits.ImageHDU` as the first argument

    Returns
    -------
    The same function decorated

    Notes
    -----
    To use this decorator
    hp_to_wcs = _hpmap(hphdu_to_wcs)
    or use @_hpmap on the function declaration
    """

    def decorator(*args, **kargs):
        """Transform a function call from (hp_map, hp_header,*) to (ImageHDU, *)"""
        if isinstance(args[0], np.ndarray) and \
           (isinstance(args[1], dict) or isinstance(args[1], fits.Header)):
            hp_hdu = fits.ImageHDU(args[0], fits.Header(args[1]))
            # Ugly fix to modigy the args
            args = list(args)
            args.insert(2, hp_hdu)
            args = tuple(args[2:])
        return hphdu_func(*args, **kargs)

    decorator._hphdu = hphdu_func
    decorator.__doc__ = update_docstring(hphdu_func, skip=6, head_docstring="""

    Parameters
    ----------
    hp_hdu : :class:`astropy.io.fits.ImageHDU`
        a pseudo ImageHDU with the healpix map and the associated header

        or

    hp_map : array_like
        healpix map with corresponding...
    hp_header : :class:`astropy.fits.header.Header`
        ...header""", foot_docstring="""
    Notes
    -----
    You can access a function using only catalogs with the ._coord() method
    """)
    return decorator
