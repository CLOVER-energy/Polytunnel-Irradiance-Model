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

from dataclasses import dataclass

import pvlib

__all__ = (
    "Location",
    "NAME",
    "NotInterceptError",
)

# NAME:
#   Keyword used for parsing the name of various components.
NAME: str = "name"


@dataclass
class Location:
    """
    Represents the location being modelled.

    .. attribute:: altitude:
        The altitude of the location.

    .. attribute:: latitude:
        The latitude of the location.

    .. attribute:: location:
        The information about the location, expressed as a :class:`pvlib.location`
        instance.

    .. attribute:: pvl

    .. attribute:: longitude:
        The longitude of the location.

    """

    altitude: float
    latitude: float
    longitude: float
    _location: pvlib.location.Location | None = None
    time_zone: str = "Europe/London"

    @property
    def location(self) -> pvlib.location.Location:
        """
        Return the information about the location in a manner that pvlib can use.

        :returns:
            A :class:`pvlib.location.Location` instance.

        """

        if self._location is None:
            self._location = pvlib.location.Location(
                self.latitude, self.longitude, tz=self.time_zone
            )

        return self._location


class NotInterceptError(Exception):
    """Raised when a vector does not intercept a plane."""
