#!/usr/bin/python3
########################################################################################
# __utils__.py --- Utility module for the irradiance model.                            #
#                                                                                      #
# Author(s): Taylor Pomfret, Emilio Nunez-Andrade, Benedict Winchester                 #
# Date created: Summer 2025                                                            #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `__utils__.py`

The model functions to compute, utilising spectral ray-tracing tools, the irradiance
distribution within a curved structure, _e.g._, a polytunnel.

"""

__all__ = (
    "MESHGRID_RESOLUTION",
    "NAME",
)

# MESHGRID_RESOLUTION:
#   Variable for storing the meshgrid resolution.
global MESHGRID_RESOLUTION
MESHGRID_RESOLUTION = 1.0

# NAME:
#   Keyword used for parsing the name of various components.
NAME: str = "name"
