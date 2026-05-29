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

import numpy as np
import pandas as pd
import pvlib

from scipy import constants

__all__ = (
    "Location",
    "NAME",
    "NotInterceptError",
    "spectrum_to_flux",
    "spectrum_to_par",
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


def spectrum_to_flux(
    spectrum: pd.Series | np.ndarray, wavelength_series: pd.Series | np.ndarray
) -> pd.Series | np.ndarray:
    """
    Convert a solar spectrum into a photon flux.

    :param: spectrum:
        The power spectrum in W/m^2.

    :param: wavelength_series:
        The wavelength data in nm.

    :returns:
        The spectrum as a photon flux spectrum in micro-moles per cm^2.

    """

    energy_series = constants.h * constants.c / (wavelength_series * (10 ** (-9)))
    if isinstance(spectrum, pd.Series):
        return (
            spectrum.divide(energy_series, axis=0)
            / (10**4 * constants.N_A)  # Convert to micro-moles per cm2
            * 10**6
        )

    return spectrum * 10**6 / (energy_series * 10**4 * constants.N_A)


def spectrum_to_par(
    spectrum: pd.Series | np.ndarray,
    par_wavelength_series: pd.Series | np.ndarray,
    wavelength_series: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """
    Convert a solar spectrum into a flux of photosynthetically-active photons (PAR).

    :param: spectrum:
        The power spectrum in W/m^2.

    :param: par_wavelength_series:
        The wavelength data in nm for the range of PAR values.

    :param: wavelength_series:
        The wavelength data in nm that match up with the spectrum.

    :returns:
        The spectrum as a photosnthetically-active photon flux spectrum in micro-moles
        per cm^2.

    """

    # Construct an energy series for the PAR wavealengths.
    energy_series = (
        constants.h * constants.c / (np.array(par_wavelength_series) * (10 ** (-9)))
    )

    # Construct a dataframe for the spectrum and clip by PAR.
    if isinstance(spectrum, pd.Series):
        raise NotImplementedError("Not implemented for pandas")

    if len(spectrum.shape) == 2:
        par_spectrum = spectrum[:, np.isin(wavelength_series, par_wavelength_series)]
    elif len(spectrum.shape) == 3:
        par_spectrum = spectrum[:, :, np.isin(wavelength_series, par_wavelength_series)]
    else:
        raise NotImplementedError(
            "Function only implemented for 2D and 3D calculations of PAR."
        )

    return par_spectrum * 10**6 / (energy_series * 10**4 * constants.N_A)
