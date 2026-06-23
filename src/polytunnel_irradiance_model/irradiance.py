########################################################################################
# irradiance.py --- Polytunnel irradiance module for the Polytunnel-Irradiance Module. #
#                                                                                      #
# Author(s): Taylor Pomfret, Emilio Nunez-Andrade, Benedict Winchester                 #
# Date created: Summer 2024/25                                                         #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `irradiance.py`

This module contains some irradiance calculations.

"""


import tracing as tracer
import os
import torch

from math import pi, sqrt
from scipy.integrate import trapezoid

import numpy as np
import pandas as pd
import tmm_fast as tmm

from src.polytunnel_irradiance_model.__utils__ import NotInterceptError
from src.polytunnel_irradiance_model.polytunnel import (
    CurveType,
    MeshPoint,
    Polytunnel,
    Vector,
)
from src.polytunnel_irradiance_model.solar import SolarPositionVector

__all__ = (
    "open_end_direct_irradiance",
    "ground_direct_irradiance",
)


def ground_direct_irradiance(
    meshpoints: MeshPoint | list[MeshPoint],
    polytunnel: Polytunnel,
    solar_irradiance: pd.Series,
    solar_position: SolarPositionVector,
    *,
    diffusivity: float | None = None,
) -> list[tuple[float, int | None]]:
    """
    Compute the direct irradiance falling on the ground.

    :param: meshpoints:
        The single :class:`Meshpoint` instance or `list` of :class:`Meshpoint` instances
        which are part of the ground mesh for which the sunlight should be computed.

    :param: polytunnel:
        The polytunnel to compute.

    :param: solar_irradiance:
        The DNI irradiance across the surface, which is simply the DNI in the beam from
        the sun, multiplied by whether a surface element is shaded.

    :param: solar_position:
        The position of the sun.

    :returns:
        - The irradiance which falls on the ground meshpoint(s) being considered'
        - The index of the meshpoint illuminating the ground.

    """

    if diffusivity is None:
        diffusivity: float = polytunnel.diffusivity

    def _single_meshpoint_ground_direct_irradiance(
        meshpoint: MeshPoint,
    ) -> tuple[float | int, int | None]:
        """
        Compute the ground irradiance falling on a single meshpoint.

        :param: meshpoint:
            The :class:`MeshPoint` instance to compute.

        :returns:
            - The direct irradiance falling on the meshpoint instance;
            - The index of the meshpoint on the surface.

        """

        # Skip if the sun is below the horizon.
        if solar_position.elevation < 0:
            return 0, None

        # Rotate everything into the polytunnel frame and translate height.
        _m = meshpoint.polytunnel_frame_position
        _m_z = _m.z + polytunnel.curve.midpoint_height

        _v = polytunnel.curve.calculate_unrotated_vector(solar_position)

        # Skip if the solar position is now below the horizontal
        if solar_position.theta_cylindrical > pi / 2:
            return 0, None

        # Compute the parameterised intercept with the surface of the polytunnel.
        _a: float = _v.x**2 + _v.z**2
        _b: float = 2 * (_m.x * _v.x + _m_z * _v.z)
        _c: float = _m.x**2 + _m_z**2 - polytunnel.curve.radius_of_curvature**2

        _discriminant: float = _b**2 - 4 * _a * _c
        # If no intercept, then return 0
        if _discriminant < 0:
            return 0, None

        _t: float = (-_b + sqrt(_discriminant)) / (2 * _a)
        _u: float = (-_b - sqrt(_discriminant)) / (2 * _a)

        # Only accept positive values
        _t_value = max(_t, _u)
        if _t_value < 0:
            raise Exception("No intercept found. Fatal error.")

        # Determine the y value for this intercept.
        _intercept_point: Vector = Vector(
            _m.x + _t_value * _v.x,
            _m.y + _t_value * _v.y,
            _m_z + _t_value * _v.z,
        )

        # If the sun did not come through the polytunnel, return 0.
        if not 0 <= _intercept_point.y <= polytunnel.length:
            return 0, None

        # Determine the meshpoint index based on this coordinate.
        try:
            surface_index, surface_point = polytunnel.surface_from_theta_y(
                _intercept_point.theta_cylindrical, _intercept_point.y
            )
        except NotInterceptError:
            return 0, None
        # un_rotated_surface_normal = polytunnel.curve.calculate_unrotated_vector(surface_point._normal_vector)

        # Multiply the irradiance on the surface by the orientation of the surface to
        # the vertical.
        return (
            solar_irradiance[surface_index]
            # The irradiance striking the ground needs to account for the orientation of
            # the sun's position;
            * _v.z * polytunnel.transmissivity * (1 - diffusivity),
            surface_index,
        )

    if not polytunnel.curve.curve_type == CurveType.CIRCULAR:
        raise NotImplementedError("Only circular curve types are implemented.")

    if isinstance(meshpoints, MeshPoint):
        return [_single_meshpoint_ground_direct_irradiance(meshpoints)]

    return [
        _single_meshpoint_ground_direct_irradiance(meshpoint)
        for meshpoint in meshpoints
    ]


def open_end_direct_irradiance(
    meshpoints: MeshPoint | list[MeshPoint],
    polytunnel: Polytunnel,
    solar_position: SolarPositionVector,
) -> list[float]:
    """
    Compute the direct irradiance that falls on the ground within the polytunnel from ends.

    If a polytunnel has open ends, then some irradiance will fall on the ground by coming in
    through these open ends. This irradiance is computed here by determining whether the
    sunlight shines in through the ends of the polytunnel onto the meshpoints.

    :param: meshpoints:
        The single :class:`Meshpoint` instance or `list` of :class:`Meshpoint` instances
        which are part of the ground mesh for which the sunlight should be computed.

    :param: polytunnel:
        The polytunnel to compute.

    :param: solar_position:
        The position of the sun.

    :returns:
        The fraction which falls on the :class:`Meshpoint` or `list` of
        :class:`Meshpoint` instances given the solar position.

    """

    # Rotate the solar position into the polytunnel frame.
    polytunnel_frame_solar_position = polytunnel.curve.calculate_unrotated_vector(
        solar_position
    )

    # Parameterise the curve with t and determine the value of t which would intersect
    # the ends of the polytunnel.
    if polytunnel_frame_solar_position.y == 0:
        return 0

    def _single_meshpoint_open_end_direct_irradiance(meshpoint: MeshPoint) -> float:
        """
        Perform the calculation for a single meshpoint.

        :param: meshpoint:
            The single :class:`Meshpoint` instance to compute for.

        :returns:
            The fraction of the irradiance which falls.

        """

        if polytunnel_frame_solar_position.y > 0:
            _t: float = (
                polytunnel.length - (_m := meshpoint.polytunnel_frame_position).y
            ) / (_v := polytunnel_frame_solar_position).y
        else:
            _t = (
                -(_m := meshpoint.polytunnel_frame_position).y
                / (_v := polytunnel_frame_solar_position).y
            )

        # Compute the x and z values for the intercept
        _x_intercept: float = _m.x + _t * _v.x
        _z_intercept: float = (
            _m_z := _m.z + polytunnel.curve.midpoint_height
        ) + _t * _v.z

        # If the intercept is out of bounds, no light fell.
        if (
            _x_intercept**2 + _z_intercept**2
        ) > polytunnel.curve.radius_of_curvature**2:
            return 0

        # If the intercept is below the horizon, no light fell.
        if _z_intercept < _m_z:
            return 0

        # Return the fraction which falls within this region.
        return _v * polytunnel.curve.calculate_unrotated_vector(meshpoint.normal_vector)

    if not polytunnel.curve.curve_type == CurveType.CIRCULAR:
        raise NotImplementedError("Only circular curve types are implemented.")

    if isinstance(meshpoints, MeshPoint):
        return [_single_meshpoint_open_end_direct_irradiance(meshpoints)]

    return [
        _single_meshpoint_open_end_direct_irradiance(meshpoint)
        for meshpoint in meshpoints
    ]
