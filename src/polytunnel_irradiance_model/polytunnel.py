########################################################################################
# polytunnel.py --- Polytunnel geomtry module for the Polytunnel-Irradiance Module.    #
#                                                                                      #
# Author(s): Taylor Pomfret, Emilio Nunez-Andrade, Benedict Winchester                 #
# Date created: Summer 2024/25                                                         #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `polytunnel.py`

This module contains all polytunnel information to define the system.

"""

import enum

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import acos, asin, atan, cos, radians, pi, sin
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np

from src.polytunnel_irradiance_model.__utils__ import NAME


# AXIS_AZIMUTH:
#   Keyword for parsing the axis azimuth information.
AXIS_AZIMUTH: str = "axis_azimuth"

# AXIS_TILT:
#   Keyword for parsing the axis tilt information.
AXIS_TILT: str = "axis_tilt"

# CURVE:
#   Keyword for parsing curve information.
CURVE: str = "curve"

# CURVE_TYPE:
#   Keyword for parsing curve type information.
CURVE_TYPE: str = "curve_type"

# Floating point precision:
#   The floating-point precision of the numbers to use when doing rotations.
FLOATING_POINT_PRECISION: int = 8

# LENGTH:
#   Keyword used for parsing the polytunnel length.
LENGTH: str = "length"

# MATERIALS:
#   Keyword used for parsing the materials information.
MATERIALS: str = "materials"

# MODULE_TYPE:
#   Keyword used for parsing module-type information.
MODULE_TYPE: str = "module_type"

# PV_MODULE:
#   Keyword for determining which of the PV modules is employed in the polytunnel.
PV_MODULE: str = "pv_module"

# PV_MODULE_SPACING:
#   Keyword for determining the spacing between PV modules.
PV_MODULE_SPACING: str = "module_spacing"

# WIDTH:
#   Keyword used for parsing the polytunnel width.
WIDTH: str = "width"


class UndergroundCellError(Exception):
    """Raised when a PV cell is underground."""


########################
# Vectors and Matrices #
########################


# Type variable for Curve and children.
_V = TypeVar(
    "_V",
    bound="Vector",
)


@dataclass
class Vector:
    """
    Represents a vector.

    .. attribute:: x:
        The x value.

    .. attribute:: y:
        The y value.

    .. attribute:: z:
        The z value.

    """

    x: float
    y: float
    z: float

    @classmethod
    def from_cylindrical_coordinates(
        cls,
        r: float,
        theta: float,
        axial_z: float,
    ) -> _V:
        """
        Instantiate a :class:`Vector` based on cylindrical coordinates.

        :param: r:
            The radial coordinate.

        :param: theta:
            The angular coordinate.

        :param: axial_z:
            The axial coordinate.

        """

        # Return re-mapped coordinates where y is along the axis and z vertical.
        y = axial_z

        # Compute x and z values.
        x = r * sin(theta)
        z = r * cos(theta)

        return cls(x, y, z)

    def __abs__(self) -> float:
        """
        Return the length of the vector.

        :returns:
            The length of the vector.

        """

        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other) -> _V:
        """
        Add two vectors together.

        :param: other:
            The vector to add to this one.

        :returns:
            The vector addition of the two vectors.

        """

        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __iadd__(self, other) -> None:
        """
        Add two vectors together.

        :param: other:
            The vector to add to this one.

        """

        self.x = self.x + other.x
        self.y = self.y + other.y
        self.z = self.z + other.z

    def __sub__(self, other) -> _V:
        """
        Subtract two vectors.

        :param: other:
            The vector to subtract from this one.

        :returns:
            The vector subtraction of the two vectors.

        """

        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __isub__(self, other) -> None:
        """
        Subtract two vectors.

        :param: other:
            The vector to subtract from this one.

        """

        self.x = self.x - other.x
        self.y = self.y - other.y
        self.z = self.z - other.z

    def __getitem__(self, key) -> float:
        """
        Allow for indexing of the vector.

        :param: key:
            The key to use to index the vector.

        :returns:
            The value from the vector.

        """

        match key:
            case 0:
                return self.x
            case 1:
                return self.y
            case 2:
                return self.z
            case _:
                raise IndexError("Vector index out of range")

    def __setitem__(self, key, value) -> None:
        """
        Allow for indexing of the vector.

        :param: key:
            The key to use to index the vector.

        :returns:
            The value from the vector.

        """

        match key:
            case 0:
                self.x = value
            case 1:
                self.y = value
            case 2:
                self.z = value
            case _:
                raise IndexError("Vector index out of range")

    def __delitem__(self, *args) -> None:
        """
        Should not be utilised as non-Physical.

        :raises:
            :class:`NotImplementedError` when called.

        """

        raise NotImplementedError("Cannot delete element from vector.")

    def __len__(self) -> int:
        """
        Return the length of the vector.

        :returns:
            The length of the vector.

        """

        return 3

    def __mul__(self, other: float | _V) -> float | _V:
        """
        Carry out scalar multiplication or dot product.

        :param: other:
            The scalar (`float`, or `int`) instance or vector to multiply with.

        :returns: Either a vector a scalar depending on what's requested.

        """

        if isinstance(other, (Vector, MeshPoint)):
            return self.x * other.x + self.y * other.y + self.z * other.z

        try:
            return Vector(self.x * other, self.y * other, self.z * other)
        except TypeError:
            raise TypeError(
                "Cannot multiply vector by non-scalar or non-vector."
            ) from None

    def __div__(self, other: float) -> _V:
        """
        Carry out scalar division.

        :param: other:
            The scalar value to divide by.

        :returns: The :class:`Vector` instance with all values rescaled.

        """

        try:
            return Vector(self.x / other, self.y / other, self.z / other)
        except TypeError:
            raise TypeError("Cannot divide vector by non-scalar or vector.") from None

    def __truediv__(self, other: float) -> _V:
        """
        Carry out scalar division.

        :param: other:
            The scalar value to divide by.

        :returns: The :class:`Vector` instance with all values rescaled.

        """

        return self.__div__(other)

    def __matmul__(self, other: _V) -> _V:
        """
        Carry out a cross product between two vectors.

        :param: other:
            The other :class:`Vector` instance to multiply with.

        :returns:
            The result of the cross product.

        """

        if not isinstance(other, Vector):
            raise NotImplementedError("Can only cross product a vector with a vector.")

        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __iter__(self) -> Iterable[float]:
        """
        Iterate over the values of the vector.

        :returns:
            An iterable instance.

        """

        return iter([self.x, self.y, self.z])

    @property
    def rows(self) -> list[float]:
        """
        Return the elements as a list.

        :returns:
            The elements as a list.

        """

        return list(self)

    @property
    def radial_coordinate(self) -> float:
        """
        Return the radial coordinate; _i.e._, the distance from the origin.

        :returns:
            The distance of the vector from the origin.

        """

        return abs(self) ** 2

    @property
    def theta(self) -> float:
        """
        Return the angle between the vector and the vertical.

        :returns:
            The angle between the vector and the vertical.

        """

        return acos(self.z / abs(self))
        # return atan(abs(self.x) / self.z) * (1 if self.x > 0 else -1)

    @property
    def phi(self) -> float | None:
        """
        Return the angle between the vector and the x axis.

        :returns:
            The angle between the vector and the x axis.

        """

        if self.x > 0:
            return atan(self.y / self.x)

        if self.x < 0:
            return atan(self.y / self.x) + (pi if self.y >= 0 else -pi)

        if self.x == 0 and self.y == 0:
            return None

        return pi / 2 if self.y > 0 else -pi / 2

    def normalise(self) -> _V:
        """
        Normalise the vector and return.

        :returns:
            The normalised vector.

        """

        self = self / abs(self)
        return self


# Type variable for Meshpoint and children.
_MP = TypeVar(
    "_MP",
    bound="MeshPoint",
)


@dataclass
class MeshPoint(Vector):
    """
    Represents a meshpoint.

    .. attribute:: area:
        The area of the mesh point.

    .. attribute:: length:
        The length of the mesh point.

    .. attribute:: width:
        The width of the mesh point.

    """

    # Private attributes:
    #
    # _corners:
    #   The `list` of corners.
    #
    # _covered_area:
    #   The area of the :class:`MeshPoint` instance that is covered with PV.
    #
    # _covered_fraction:
    #   The fraction of the :class:`MeshPoint` instance which is covered with PV.
    #
    # _normal_vector:
    #   The vector which is normal to the :class:`MeshPoint` instance.
    #

    length: float | None = None
    width: float | None = None
    _area: float | None = None
    _corners: list[Vector] | None = None
    _covered_area: float | None = None
    _covered_fraction: float | None = None
    _normal_vector: Vector | None = None
    _polytunnel_frame_position: Vector | None = None
    _u_vector: Vector | None = None
    _t_vector: Vector | None = None

    @classmethod
    def from_cylindrical_coordinates(
        cls,
        r: float,
        theta: float,
        axial_z: float,
        length: float | None = None,
        width: float | None = None,
    ) -> _MP:
        """
        Instantiate a meshpoint based on cylindrical coordinates aligned with its axis.

        :param: r:
            The radial coordinate.

        :param: theta:
            The angular coordinate.

        :param: axial_z:
            The axial coordinate.

        :param: area:
            The area of the point.

        """

        # Return re-mapped coordinates where y is along the axis and z vertical.
        y = axial_z

        # Compute x and z values.
        x = r * sin(theta)
        z = r * cos(theta)

        # Instantiate and return.
        return cls(x, y, z, length, width)

    @property
    def area(self) -> float:
        """
        Return the area of the meshpoint.

        :returns:
            The calculated area of the meshpoint.

        """

        if self._area is None:
            self._area = self.length * self.width

        return self._area

    @property
    def position_vector(self) -> Vector:
        """
        Return the position vector specifying the centre of the :class:`MeshPoint`.

        :returns:
            The centre point of the meshpoint.

        """

        return Vector(self.x, self.y, self.z)

    def __post_init__(self) -> None:
        """Method run post instantiation of the point."""

        # Compute the normal vector to the point.
        if self._normal_vector is None:
            self._normal_vector = Vector(self.x, self.y, self.z).normalise()

        # Compute the corners of the point (given that the point starts as a square).
        # Two vectors within the meshpoint can be found. The first will be parallel to
        # y and have a length which is the sqrt(area).
        _y_vector: Vector = Vector(0, self.length / 2, 0)

        # The x vector will simply be the cross product of this, normalised, and
        # rescaled based on the length in that direction.
        _x_vector: Vector = self._normal_vector @ _y_vector
        _x_vector = _x_vector.normalise() * self.width / 2

        # Compute the four corners.
        self._corners: list[Vector] = [
            self.position_vector + _x_vector + _y_vector,
            self.position_vector + _x_vector - _y_vector,
            self.position_vector - _x_vector + _y_vector,
            self.position_vector - _x_vector - _y_vector,
        ]

        # Save the u and t vectors.
        self._t_vector = _x_vector
        self._u_vector = _y_vector

        # Save a vector representing the position of the meshpoint but within the frame
        # of the polytunnel.
        self._polytunnel_frame_position = Vector(self.x, self.y, self.z)

    @property
    def covered_area(self) -> float:
        """
        Return the covered area of the :class:`MeshPoint` instance.

        :returns:
            The area of the mesh point that is covered with PV.

        """

        if self._covered_area is None:
            raise Exception("Covered area should be set before being called.")

        return self._covered_area

    @property
    def covered_fraction(self) -> float:
        """
        Return, and compute, if necessary, the fraction of the mesh point covered.

        :returns:
            The fraction of the meshpoint covered with PV.

        """

        if self._covered_fraction is None:
            self._covered_fraction = self.covered_area / self.area

        return self._covered_fraction

    def set_normal_vector(self, new_normal_vector: Vector) -> None:
        """
        Set the normal vector on the meshpoint, _e.g._, after an operation.

        :param: new_normal_vector:
            The new normal vector to use.

        """

        self._normal_vector = new_normal_vector


# Type variable for Curve and children.
_M = TypeVar(
    "_M",
    bound="Matrix",
)


@dataclass
class Matrix:
    """
    Represents a matrix.

    .. attribute:: array:
        The matrix points.

    """

    _array: list[list[float]]
    _transpose: list[list[float]] | None

    def __post_init_(self) -> None:
        """Carry out post-instantiation checks on the matrix."""

        if len(row_lengths := {len([row for row in self._array])}) != 1:
            raise Exception("Matrix instances need to have rows of equal length.")

        if list(row_lengths)[0] != 3:
            raise Exception(
                "Matrix should have rows of length 3 for use in 3D geometry."
            )

        # Compute and save the transpose of the Matrix elements.
        self._transpose = [
            [row[column_index] for row in self._array]
            for column_index in range(len(self._array))
        ]

    @property
    def transpose(self) -> _M:
        """
        Return the transpose of the matrix.

        :returns:
            The transpose of the matrix.

        """

        self._array = [
            [row[column_number] for row in self.rows] for column_number in range(3)
        ]
        return self

    @property
    def columns(self) -> list[Vector]:
        """
        Return the columns contained within the matrix.

        :returns:
            The columns within the matrix.

        """

        return [
            Vector(*[row[column_number] for row in self.rows])
            for column_number in range(3)
        ]

    @property
    def rows(self) -> list[Vector]:
        """
        Return the rows contained within the matrix.

        :returns:
            The rows within the matrix.

        """

        return [Vector(*row) for row in self._array]

    def __matmul__(self, other: _M | Vector | MeshPoint) -> _M | Vector | MeshPoint:
        """
        Carry out matrix multiplication of the current matrix.

        :param: other:
            The :class:`Matrix` or :class:`Vector` instance to multiply with.

        :raises: NotImplementedError:
            Raised if matrix multiplication is attempted on a MeshPoint instance.

        :return:
            The result of the matrix multiplication.

        """

        # Check that matrix multiplication can be carried out.
        if len(self.columns) != len(other.rows):
            raise Exception("Matrix dimensions do not match.")

        # If multiplying by a vector, simply return the new vector.
        if isinstance(other, Vector):
            return Vector(*[row * other for row in self.rows])

        # If multiplying by a meshpoint, then raise an error as this should only be
        # implemented in children.
        if isinstance(other, MeshPoint):
            raise NotImplementedError(
                "Matrix multiplication must use children of Matrix."
            )

        # Otherwise, carry out matrix multiplication.
        new_rows: list[Vector] = []
        for row in self.rows:
            new_rows.append([row * column for column in other.columns])

        return Matrix([list(row) for row in new_rows], None)


# Type variable for RotationMatrix and children.
_DM = TypeVar(
    "_DM",
    bound="DiagonalMatrix",
)


@dataclass
class DiagonalMatrix(Matrix):
    """
    Represents a diagonalised matrix.

    .. attribute:: diagonal:
        The diagonal of the matrix.

    """

    diagonal: Vector

    @classmethod
    def from_diag(cls, diagonal: Vector) -> _DM:
        """
        Instantiate a :class:`DiagonalMatrix` instance from its diagonal.

        :param: diagonal:
            The diagonal to use.

        :returns:
            An instantiated :class:`DiagonalMatrix` instance.

        """

        return cls(
            [
                [
                    diagonal[column_index] if column_index == row_index else 0
                    for column_index in range(len(diagonal))
                ]
                for row_index in range(len(diagonal))
            ],
            None,
            diagonal,
        )

    def __matmul__(self, other: _M | Vector | MeshPoint) -> _M | Vector | MeshPoint:
        """
        Carry out matrix multiplication of the current matrix.

        :param: other:
            The :class:`Matrix` or :class:`Vector` instance to multiply with.

        :return:
            The result of the matrix multiplication.

        """

        if isinstance(other, MeshPoint):
            if len({entry != 0 for entry in self.diag}) != 1:
                raise NotImplementedError(
                    "Cannot stretch a Meshpoint if not oriented correctly."
                )

            other.x *= self.diagonal[0]
            other.y *= self.diagonal[1]
            other.z *= self.diagonal[2]

            return other

        super().__matmul__(other)


class CartesianAxis(enum.Enum):
    """
    Denotes a Cartesian axis.

    - X:
        The x-axis.

    - Y:
        The y-axis.

    - Z:
        The z-axis.

    """

    X: str = "x"
    Y: str = "y"
    Z: str = "z"


# Type variable for RotationMatrix and children.
_RM = TypeVar(
    "_RM",
    bound="RotationMatrix",
)


@dataclass
class RotationMatrix(Matrix):
    """
    Represents a rotation matrix.

    .. attribute:: rotation_angle:
        The angle, in radians, for the rotation.

    .. attribute:: rotation_axis:
        The axis (x, y, or z) about which the rotation occurs.

    """

    rotation_angle: float
    rotation_axis: CartesianAxis

    @classmethod
    def from_rotation_angle_and_axis(
        cls, rotation_angle: float, rotation_axis: CartesianAxis
    ) -> _RM:
        """
        Instnatiate a rotation matrix based on the rotation angle and axis.

        :param: rotation_angle:
            The angle for the rotation, in radians.

        :param: rotation_axis:
            The axis around which the rotation occurs.

        """

        # Compute the rotation matrix based on the angle and axis.
        match rotation_axis:
            case CartesianAxis.X:
                array: list[list[float]] = [
                    [1, 0, 0],
                    [
                        0,
                        cos(rotation_angle),
                        -sin(rotation_angle),
                    ],
                    [
                        0,
                        sin(rotation_angle),
                        cos(rotation_angle),
                    ],
                ]
            case CartesianAxis.Y:
                array: list[list[float]] = [
                    [
                        cos(rotation_angle),
                        0,
                        sin(rotation_angle),
                    ],
                    [0, 1, 0],
                    [
                        -sin(rotation_angle),
                        0,
                        cos(rotation_angle),
                    ],
                ]
            case CartesianAxis.Z:
                array: list[list[float]] = [
                    [
                        cos(rotation_angle),
                        -sin(rotation_angle),
                        0,
                    ],
                    [
                        sin(rotation_angle),
                        cos(rotation_angle),
                        0,
                    ],
                    [0, 0, 1],
                ]
            case _:
                raise Exception("Internal error, check rotation matrix instantiation.")

        return cls(array, None, rotation_angle, rotation_axis)

    def __matmul__(self, other: _M | Vector | MeshPoint) -> _M | Vector | MeshPoint:
        """
        Carry out matrix multiplication of the current matrix.

        :param: other:
            The :class:`Matrix` or :class:`Vector` instance to multiply with.

        :return:
            The result of the matrix multiplication.

        """

        # If multiplying by a meshpoint, then rotate the meshpoint and its normal.
        if isinstance(other, MeshPoint):
            coordinate_vector = Vector(*[row * other for row in self.rows])
            normal_vector = Vector(*[row * other._normal_vector for row in self.rows])

            other.x, other.y, other.z = (
                coordinate_vector.x,
                coordinate_vector.y,
                coordinate_vector.z,
            )
            other.set_normal_vector(normal_vector)

            return other

        return super().__matmul__(other)


class ModuleType(enum.Enum):
    """
    Denotes the type of the module.

    - CRYSTALINE:
        Used to distinguish crystaline Silicon modules.

    - THIN_FILM:
        Used to distinguish thin-film modules.

    """

    CRYSTALINE: str = "c_Si"
    CIGS: str = "cigs"
    THIN_FILM: str = "thin_film"


############
# PVModule #
############


# Type variable for Polytunnel and children.
_PVCM = TypeVar(
    "_PVCM",
    bound="PVCellMaterial",
)


@dataclass
class PVCellMaterial:
    """
    Represents a layer within the PV module.

    .. attribute:: name:
        The name of the layer.

    .. attribute:: thickness:
        The thickness of the layer.

    """

    name: str
    thickness: float

    @classmethod
    def from_entry(cls, entry: dict[str, float]) -> _PVCM:
        """
        Instantiate and return a :class:`PVCellMaterial` instance based on the entry.

        :param: entry:
            The entry within the mapping.

        :returns: The instantiated instance.

        """

        return cls(name=list(entry.keys())[0], thickness=list(entry.values())[0])


@dataclass(kw_only=True)
class PVModule:
    """
    Contains information about the PV module.

    .. attribute:: length:
        The length of the module, in meters.

    .. attribute:: materials:
        The materials included, along with their thicknesses.

    .. attribute:: module_type:
        The type of the module.

    .. attribute:: multistack:
        The number of stacks to include


    .. attribute:: name:
        The name of the :class:`PVModule` instance.

    .. attribute:: orientation:
        The angle between the axis of the polytunnel and of the module.

    .. attribute:: width:
        The width of the cell, measured in metres.

    """

    length: float
    materials: list[PVCellMaterial]
    module_type: ModuleType
    multistack: int | None
    name: str
    orientation: float
    width: float


#########
# Curve #
#########


class CurveType(enum.Enum):
    """
    Denotes the type of curve. Useful in constructing curves.

    - CIRCULAR:
        Denotes a circular geometry.

    - ELIPTICAL:
        Denotes an eliptical geometry.

    """

    CIRCULAR: str = "circular"
    ELIPTICAL: str = "eliptical"


# Type variable for Curve and children.
_C = TypeVar(
    "_C",
    bound="Curve",
)

# TYPE_TO_CURVE_MAPPING:
#   Mapping between the curve type and curve instances.
TYPE_TO_CURVE_MAPPING: dict[CurveType, _C] = {}


@dataclass(kw_only=True)
class Curve(ABC):
    """
    Represents a curve around which the PV module curves.

    A curved surface, _e.g._, a polytunnel, has a given axis around which it curves (in
    the case of a polytunnel, this is its length) an equation for its curve (which may
    be a simple circle or a more complex shape like a parabola or hyperbola) and a
    length scale. These last two are wrapped up in a single function which takes a
    disaplcement along the curve and returns the angles at that point.

    .. attribute:: curvature_azimuth:
        The azimuth angle for the curvature axis in degrees.

    .. attribute:: curvature_tilt:
        The tilt angle for the curvature axis in degrees.

    .. attribute:: get_angles_from_surface_disaplacement:
        A callable function which can return the azimuth and tilt at any point on the
        surface based on the distance from the central axis.

    """

    axis_azimuth: float = 180
    axis_tilt: float = 0
    name: str = ""
    width: float | None = None
    _axial_vector: Vector | None = None
    _azimuth_rotation_matrix: list[list[float]] | None = None
    _maximum_arc_length: float | None = None
    _midpoint_height: float = 0
    _tilt_rotation_matrix: list[list[float]] | None = None
    _vertical_vector: Vector | None = None

    def __init_subclass__(cls, curve_type: CurveType) -> None:
        """
        Hook used to store the type of the curve.

        :param: **curve_type:**
            The type of the curve.

        """

        cls.curve_type = curve_type  # type: ignore [attr-defined]
        TYPE_TO_CURVE_MAPPING[curve_type] = cls

        super().__init_subclass__()

    @property
    def axial_vector(self) -> Vector:
        """
        Return the axial vector.

        :returns:
            The axial vector.

        """

        if self._axial_vector is None:
            raise Exception("Axial vector called before calculated.")

        return self._axial_vector

    @property
    def vertical_vector(self) -> Vector:
        """
        Return the vertical vector.

        :returns:
            The vertical vector.

        """

        if self._vertical_vector is None:
            raise Exception("Axial vector called before calculated.")

        return self._vertical_vector

    @property
    def midpoint_height(self) -> float:
        """
        Return the height of the midpoint above or below the zero offset value.

        :returns:
            The height of the midpoint of the curve.

        """

        return self._midpoint_height

    def __post_init__(self) -> None:
        """
        Post-instantiation method to compute axial and vertical vectors.

        These vectors point along the axis of the curve and vertically, respectively,
        and are calculated based on these vectors followed by rotation matrices.

        """

        self._axial_vector = self._calculate_rotated_vector(Vector(0, 1, 0))
        self._vertical_vector = self._calculate_rotated_vector(Vector(0, 0, 1))

    def _instantiate_mesh(
        self, length: float, meshgrid_resolution: int
    ) -> Iterator[MeshPoint]:
        """
        Generate a series of points around the curve, un-distorted.

        A radius of 1 m is used as default, with the resolution specifying the number of
        points along the polytunnel and around it.

        :param: length:
            The length of the polyertunnel.

        :param: meshgrid_resolution:
            The number of points to use in each direction.

        :returns: A `list` of :class:`MeshPoint` instances.

        """

        # Set a default radius for the un-distorted shape.
        radius: int = 1

        # Determine the radial distance transcribed by a single meshpoint.
        # This value will be the angule transcribed, in radians, multiplied by the
        # radius of curvature of the curve.
        meshpoint_width = radius * self.maximum_theta_value / (meshgrid_resolution / 2)

        # Setup theta and z iterators.
        for theta in np.linspace(
            -(self.maximum_theta_value - (angular_size := pi / meshgrid_resolution)),
            self.maximum_theta_value - angular_size,
            meshgrid_resolution,
        ):
            for z in np.linspace(0, length, meshgrid_resolution):
                yield MeshPoint.from_cylindrical_coordinates(
                    radius, theta, z, length / meshgrid_resolution, meshpoint_width
                )

    def _mesh_overlap(
        self, meshgrid: list[MeshPoint], pv_module: PVModule, pv_module_spacing: float
    ) -> list[MeshPoint]:
        """
        Update the mesh points with the area and fractional area that overlaps modules.

        :param: meshgrid:
            The `list` of :class:`MeshPoint` instances.

        :param: PVModule:
            The :class:`PVModule` instance.

        :param: pv_module_spacing:
            The spacing between PV modules.

        :returns:
            The updated `list` of :class:`MeshPoint` instances where each's overlap with
            the PV modules on the polytunnel is computed.

        """

        # TODO: Implement this function.

        return meshgrid

    @abstractmethod
    def _stretch_mesh(self, meshgrid: list[MeshPoint]) -> list[MeshPoint]:
        """
        Stretch/distort the meshgrid to match the geometry of the curve.

        :param: meshgrid:
            The meshgrid, as a `list` of :class:`MeshPoint` instances, to stretch.

        :returns: The distorted `list` of :class:`MeshPoint` instances.

        """

        raise NotImplementedError(
            "Cannot call `_stretch_mesh` method on abstract curve."
        )

    @abstractmethod
    def _realign_mesh(self, meshgrid: list[MeshPoint]) -> list[MeshPoint]:
        """
        Reset the meshgrid so that the polytunnel reaches the ground.

        :param: meshgrid:
            The meshgrid, as a `list` of :class:`MeshPoint` instances, to stretch.

        :returns: The reliagned  `list` of :class:`MeshPoint` instances.

        """

        raise NotImplementedError(
            "Cannot call `_realign_mesh` method on abstract curve."
        )

    def rotate_mesh(self, meshgrid: list[MeshPoint]) -> list[MeshPoint]:
        """
        Rotate the meshgrid points and normal vectors according based on the curve axis.

        :param: meshgrid:
            The meshgrid, as a `list` of :class:`MeshPoint` instances, to rotate.

        :returns: The rotated `list` of :class:`MeshPoint` instances.

        """

        return [self._calculate_rotated_vector(meshpoint) for meshpoint in meshgrid]

    @property
    @abstractmethod
    def maximum_theta_value(self) -> float:
        """
        The maximum arc length of the curve, measured in radians.

        :returns:
            The maximum arc length of the curve, measured in radians.

        """

        raise NotImplementedError("Method should be implemented in child classes.")

    def generate_mesh(
        self,
        length: float,
        meshgrid_resolution: int,
        pv_module: PVModule,
        pv_module_spacing: float,
    ) -> list[MeshPoint]:
        """
        Generate the mesh for the curve.

        Generation of the mesh will depend on the exact curve shape which is implemented
        and so is here listed as an abstract method.

        The process, in general, will consist of:
        1. Generating an equally-spaced grid of mesh points on the surface of the curve;
        2. Distorting these points, where appropriate, based on the geometry of the
           curve;
        3. Computing the overal of the PV modules with the mesh points;
        4. Rotating all mesh points and normal vectors.

        :param: length:
            The length of the Polytunnel.

        :param: meshgrid_resolution:
            The number of meshgrid points to use.

        :param: pv_module:
            The :class:`PVModule` instance.

        :param: pv_module_spacing:
            The axial spacing between PV modules.

        :returns:
            The instantiated mesh.

        """

        # Generate an equally-spaced, un-distorted mesh.
        meshgrid = list(self._instantiate_mesh(length, meshgrid_resolution))

        # Stretch the meshgrid.
        meshgrid = self._stretch_mesh(meshgrid)

        # Re-align the mesh so that the points are on the ground, as appopriate.
        meshgrid = self._realign_mesh(meshgrid)

        # Compute the overlap of the mesh with the PV modules.
        meshgrid = self._mesh_overlap(meshgrid, pv_module, pv_module_spacing)

        # Rotate all meshpoints
        return self.rotate_mesh(meshgrid)

    @property
    def zenith_rotation_angle(self) -> float:
        """
        The rotation about the zenith is not the azimuth this is calculated here.

        A user-specified azimuth angle gives and angle from North (y=0) which is the
        opposite to a rotation carried out.

        """

        return -self.axis_azimuth

    @property
    def azimuth_rotation_matrix(self) -> list[list[float]]:
        """
        The rotation matrix for an azimuth rotation.

        Returns:
            - A `list` of `list`s representing the matrix.

        """

        if self._azimuth_rotation_matrix is None:
            self._azimuth_rotation_matrix = RotationMatrix.from_rotation_angle_and_axis(
                self.zenith_rotation_angle, CartesianAxis.Z
            )

        return self._azimuth_rotation_matrix

    @property
    def tilt_rotation_angle(self) -> float:
        """
        The rotation angle for the zenith rotation needs to be adjusted also.

        Rotations about the x axis result in a tilting in a southerly rather than a
        northerly direction. This is adjusted for here.

        """

        return -self.axis_tilt

    @property
    def tilt_rotation_matrix(self) -> list[list[float]]:
        """
        The rotation matrix for an azimuth rotation.

        Returns:
            - A `list` of `list`s representing the matrix.

        """

        if self._tilt_rotation_matrix is None:
            self._tilt_rotation_matrix = RotationMatrix.from_rotation_angle_and_axis(
                self.tilt_rotation_angle, CartesianAxis.X
            )

        return self._tilt_rotation_matrix

    def _calculate_rotated_vector(self, un_rotated_normal: MeshPoint) -> MeshPoint:
        """
        Rotate the vector based on the orientation of the curve/Polytunnel.

        :param: **un_rotated_normal:**
            The surface or position vector in the un-rotated frame.

        :returns: A :class:`MeshPoint` instance.

        """

        # Rotate this normal vector based on the tilt and azimuth of the polytunnel.
        rotated_normal = (
            self.azimuth_rotation_matrix @ self.tilt_rotation_matrix @ un_rotated_normal
        )

        return rotated_normal


@dataclass(kw_only=True)
class CircularCurve(Curve, curve_type=CurveType.CIRCULAR):
    """
    Represents a circular geometry. In this instance, a radius of curvature is required.

    .. attribute:: radius_of_curvature:
        Represents the radius of curvature.

    """

    radius_of_curvature: float

    @property
    def maximum_theta_value(self) -> float:
        """
        The maximum arc length of the curve, measured in radians.

        :returns:
            The maximum arc length of the curve, measured in radians.

        """

        if self._maximum_arc_length is None:
            # Determine the maximum angular limits of the curve based on the chord width.
            try:
                theta_max = asin(
                    (width_to_diameter := self.width / (2 * self.radius_of_curvature))
                )
            except ValueError:
                if width_to_diameter > 1:
                    raise ValueError(
                        "Width of the polytunnel must not be greater than its diameter."
                    ) from None
                else:
                    raise

            self._maximum_arc_length = theta_max

        return self._maximum_arc_length

    def _stretch_mesh(self, meshgrid: list[MeshPoint]) -> list[MeshPoint]:
        """
        Stretch/distort the meshgrid to match the geometry of the curve.

        For a circular curve, this involves moving all of the points out radially. A
        solution to this is to consider each radial stretch as a rotation to be aligned
        with the z-axis, a stretch in the z-direction, and then a rotation back to the
        original position of the point.

        This requires
        - rotating by the theta angle of the point;
        - stretching in the z direction;
        - then rotating back.

        Because each angle is different, the matrix for each point needs calculating
        individually.

        :param: meshgrid:
            The meshgrid, as a `list` of :class:`MeshPoint` instances, to stretch.

        :returns: The distorted `list` of :class:`MeshPoint` instances.

        """

        # Construct stretch matrices.
        z_angles = [vector.theta for vector in meshgrid]
        stretch_matrices = [
            RotationMatrix.from_rotation_angle_and_axis(theta, CartesianAxis.Y)
            @ DiagonalMatrix.from_diag(Vector(1, 1, self.radius_of_curvature))
            @ RotationMatrix.from_rotation_angle_and_axis(-theta, CartesianAxis.Y)
            for theta in z_angles
        ]

        # Carry out the stretching.
        return [
            stretch_matrices[index] @ meshpoint
            for index, meshpoint in enumerate(meshgrid)
        ]

    def _realign_mesh(self, meshgrid: list[MeshPoint]) -> list[MeshPoint]:
        """
        Reset the meshgrid so that the polytunnel reaches the ground.

        :param: meshgrid:
            The meshgrid, as a `list` of :class:`MeshPoint` instances, to stretch.

        :returns: The reliagned  `list` of :class:`MeshPoint` instances.

        """

        self._midpoint_height = self.radius_of_curvature * cos(self.maximum_theta_value)

        return [
            meshpoint - Vector(0, 0, self._midpoint_height) for meshpoint in meshgrid
        ]


@dataclass(kw_only=True)
class ElipticalCurve(Curve, curve_type=CurveType.ELIPTICAL):
    """
    Represents a eliptical geometry. In this instance, semi-major and -minor axes.

    .. attribute:: semi_major_axis:
        The semi-major axis of the elipse.

    .. attribute:: semi_minor_axis:
        The semi-minor axis of the elipse.

    """

    semi_major_axis: float
    semi_minor_axis: float

    def __post_init__(self) -> None:
        """Post-instantiation script."""

        raise NotImplementedError("Eliptical curves are not implemented.")


# Type variable for Polytunnel and children.
_P = TypeVar(
    "_P",
    bound="Polytunnel",
)


class Polytunnel:
    """
    Represents a Polytunnel instance.

    .. attribute:: curve:
        The curve which defines the shape of the polytunnel.

    .. attribute:: ground_mesh:
        The mesh of :class:`MeshPoint` instances representing the ground within the
        :class:`Polytunnel` instance.

    .. attribute:: length:
        The length of the polytunnel.

    .. attribute:: meshgrid_resolution:
        The resolution of meshgrid, in meters.

    .. attribute:: name:
        The name of the polytunnel instance.

    .. attribute:: pv_module:
        The pv module on the polytunnel.

    .. attribute:: pv_module_spacing:
        The spacing between the PV modules.

    .. attribute:: surface_mesh:
        The mesh of :class:`MeshPoint` instances representing the surface of the
        :class:`Polytunnel` instance.

    """

    def __init__(
        self,
        curve: Curve,
        length: float,
        meshgrid_resolution: int,
        name: str,
        pv_module: PVModule,
        pv_module_spacing: float,
    ):
        """
        Instantiate a Polytunnel instance.

        :param: length:
            The length of the polytunnel.

        :param: meshgrid_resolution
            The resolution of the meshgrid for the polytunnel.

        """

        self.curve = curve
        self.length = length
        self.meshgrid_resolution = meshgrid_resolution
        self.name = name
        self.pv_module = pv_module
        self.pv_module_spacing = pv_module_spacing

        # Generate and store the meshes on the instance of the polytunnel.
        self.ground_mesh: list[MeshPoint] = self.generate_ground_mesh()
        self.surface_mesh: list[MeshPoint] = self.curve.generate_mesh(
            self.length, meshgrid_resolution, self.pv_module, self.pv_module_spacing
        )

    @property
    def axial_vector(self) -> Vector:
        """
        Return the axial vector.

        :returns:
            The axial vector.

        """

        return self.curve.axial_vector

    @property
    def vertical_vector(self) -> Vector:
        """
        Return the vertical vector.

        :returns:
            The vertical vector.

        """

        return self.curve.vertical_vector

    @property
    def width(self) -> float:
        """
        The width of the polytunnel.

        This value is the same as the width of the curve.

        :returns:
            The width of the polytunnel.

        """

        return self.curve.width

    def generate_ground_mesh(self) -> list[MeshPoint]:
        """
        Generate the set of points representing the ground within the polytunnel.

        This process consists of:
        - creating equally spaced points within the polytunnel along the ground;
        - rotating and tilting these as necessary to match its orientation.

        """

        # Create a mesh within the Polytunnel instance.
        meshgrid: list[MeshPoint] = []
        for x in np.linspace(-self.width / 2, self.width / 2, self.meshgrid_resolution):
            for y in np.linspace(0, self.length, self.meshgrid_resolution):
                meshgrid.append(
                    MeshPoint(
                        x,
                        y,
                        0,
                        self.length / self.meshgrid_resolution,
                        self.width / self.meshgrid_resolution,
                        _normal_vector=Vector(0, 0, 1),
                    )
                )

        # Rotate these points as appropriate.
        return self.curve.rotate_mesh(meshgrid)

    @classmethod
    def from_data(
        cls,
        input_data: dict[str, Any],
        meshgrid_resolution: int,
        module_input_data: dict[str, Any],
    ) -> _P:
        """
        Instantiate a :class:`Polytunnel` instance based on the input data provided.

        :param: input_data:
            The input data.

        :param: meshgrid_resolution:
            The resolution of the meshgrid.

        :param: module_input_data:
            The input data about the PV module.

        :returns: An instantiated :class:`Polytunnel` instance.

        """

        # Parse the curve information.
        input_data[CURVE][AXIS_AZIMUTH] = radians(input_data[CURVE][AXIS_AZIMUTH])
        input_data[CURVE][AXIS_TILT] = radians(input_data[CURVE][AXIS_TILT])

        # Instantiate the curve.
        try:
            curve = TYPE_TO_CURVE_MAPPING[CurveType(input_data[CURVE].pop(CURVE_TYPE))](
                **input_data[CURVE]
            )
        except KeyError:
            raise KeyError("Missing curve information.") from None

        # Parse the module information.
        try:
            module_input_data[module_name][MODULE_TYPE] = ModuleType(
                module_input_data[(module_name := input_data[PV_MODULE])][MODULE_TYPE]
            )
        except KeyError:
            raise KeyError("Missing module-type information.") from None

        try:
            module_input_data[module_name][MATERIALS] = [
                PVCellMaterial.from_entry(material_information)
                for material_information in module_input_data[module_name][MATERIALS]
            ]
        except KeyError:
            raise KeyError("Missing PV material information.") from None

        try:
            pv_module = PVModule(**module_input_data[input_data[PV_MODULE]])
        except KeyError:
            raise KeyError("Missing PV-module information.") from None

        return cls(
            curve,
            input_data[LENGTH],
            meshgrid_resolution,
            input_data[NAME],
            pv_module,
            input_data[PV_MODULE_SPACING],
        )

    # class ElipticalPolytunnel(PolytunnelShape.ELIPTICAL):
    #     """
    #     Represents an eliptical polytunnel.

    #     .. attribute:: semi_major_axis:
    #         The semi-major axis of the polytunnel.

    #     .. attribute:: semi_minor_axis:
    #         The semi-minor axis of the polytunnel.

    #     """

    #     def __init__(
    #         self,
    #         azimuthal_orientation: float,
    #         length: float,
    #         pv_module: PVModule,
    #         semi_major_axis: float,
    #         semi_minor_axis: float,
    #         tilt: float,
    #         width: float,
    #         *,
    #         meshgrid_resolution=1.0,
    #     ):
    #         """
    #         Instantiate an eliptical polytunnel instance.

    #         :param: semi_major_axis
    #             The semi-major axis of the polytunnel.

    #         :param: semi_minor_axis
    #             The semi-minor axis of the polytunnel.

    #         """

    #         # Enforce that the elipse is not a circle.
    #         if semi_minor_axis == semi_major_axis:
    #             raise Exception(
    #                 "Instantiation of an eliptical polytunnel with circular arguments."
    #             )

    #         self.semi_major_axis = semi_major_axis
    #         self.semi_minor_axis = semi_minor_axis

    #         super().__init__(
    #             azimuthal_orientation, length, meshgrid_resolution, pv_module, tilt, width
    #         )

    # def apply_rotations(self, X, Y, Z):
    #     """
    #     Appy two rotations:
    #     - First: Around Z by azimuthal_orientation (in radians).
    #     - Second: Around X by tilt  (in radians).
    #     """
    #     # Rotation around Z
    #     R_z = np.array(
    #         [
    #             [
    #                 np.cos(self.azimuthal_orientation),
    #                 -np.sin(self.azimuthal_orientation),
    #                 0,
    #             ],
    #             [
    #                 np.sin(self.azimuthal_orientation),
    #                 np.cos(self.azimuthal_orientation),
    #                 0,
    #             ],
    #             [0, 0, 1],
    #         ]
    #     )

    #     # Rotation around X
    #     R_x = np.array(
    #         [
    #             [1, 0, 0],
    #             [0, np.cos(self.tilt), -np.sin(self.tilt)],
    #             [0, np.sin(self.tilt), np.cos(self.tilt)],
    #         ]
    #     )

    #     # Flat coordinates to make an easier rotation
    #     coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    #     # Apply rotations sequentially
    #     rotated_coords = R_z @ (R_x @ coords)

    #     # Reshape coordinates to keep same as input X,Y,Z arrays
    #     X_rot, Y_rot, Z_rot = rotated_coords.reshape(3, *X.shape)
    #     return X_rot, Y_rot, Z_rot

    # def generate_ground_grid(self):
    #     # y_ground = np.linspace(-self.length, self.length, self.n)
    #     y_ground = np.linspace(0, self.length, self.length_wise_mesh_resolution)
    #     x_ground = np.linspace(
    #         -self.semi_major_axis + 0.0001,
    #         self.semi_major_axis - 0.0001,
    #         self.n_angular,
    #     )

    #     X, Y = np.meshgrid(x_ground, y_ground)
    #     Z = np.zeros_like(X)

    #     # MODIFICATION
    #     # Rotation in the XY plane
    #     # cos_xy = np.cos(self.tilt)
    #     # sin_xy = np.sin(self.tilt)

    #     # X_rot_xy = X * cos_xy - Y * sin_xy
    #     # Y = X * sin_xy + Y * cos_xy

    #     # # Rotation around the Z-axis
    #     # cos_z = np.cos(self.azimuthal_orientation)
    #     # sin_z = np.sin(self.azimuthal_orientation)

    #     # X = X_rot_xy * cos_z - Z * sin_z
    #     # Z = X_rot_xy * sin_z + Z * cos_z  # Z_final will still be zero since Z is zero initially

    #     if self.tilt != 0 or self.azimuthal_orientation != 0:
    #         X, Y, Z = self.apply_rotations(X, Y, Z)

    #     X = X + self.x_shift
    #     Y = Y + self.y_shift
    #     Z = Z + self.z_shift

    #     return (X, Y, Z)

    # def generate_surface(self):
    #     """
    #     The parametric definition of this cylinder surface (S) is:
    #        S = [
    #            semi_major_axis * cos(theta),
    #            length,
    #            semi_minor_axis * sin(theta)
    #        ],
    #     """

    #     # Create the grid in cylindrical coordinates
    #     # y = np.linspace(-self.length, self.length, self.n)
    #     y = np.linspace(0, self.length, self.length_wise_mesh_resolution)
    #     theta = np.linspace(0, np.pi, self.n_angular)  # Semi-circular cross-section

    #     # Create a meshgrid for Y and Theta
    #     Y, Theta = np.meshgrid(y, theta)

    #     # Convert cylindrical coordinates to Cartesian coordinates
    #     X = self.semi_major_axis * np.cos(Theta)  # Horizontal width
    #     Z = self.semi_minor_axis * np.sin(Theta)  # Height above the ground

    #     # MODIFICATION
    #     # # Rotation in the XY plane
    #     # cos_xy = np.cos(self.tilt)
    #     # sin_xy = np.sin(self.tilt)

    #     # X_rot_xy = X * cos_xy - Y * sin_xy
    #     # Y = X * sin_xy + Y * cos_xy

    #     # # Rotation around the Z-axis
    #     # cos_z = np.cos(self.azimuthal_orientation)
    #     # sin_z = np.sin(self.azimuthal_orientation)

    #     # X = X_rot_xy * cos_z - Z * sin_z
    #     # Z = X_rot_xy * sin_z + Z * cos_z
    #     if self.tilt != 0 or self.azimuthal_orientation != 0:
    #         X, Y, Z = self.apply_rotations(X, Y, Z)

    #     X = X + self.x_shift
    #     Y = Y + self.y_shift
    #     Z = Z + self.z_shift

    #     # Parameters for solar cells
    #     theta_margin = (
    #         self.theta_margin
    #     )  # Margin on the left and right edges (in terms of segments)
    #     cell_thickness = (
    #         self.cell_thickness
    #     )  # Thickness of the solar cell (in terms of segments)
    #     cell_spacing = (
    #         self.cell_spacing
    #     )  # Gap between solar cells (in terms of segments)

    #     # Initialize an array to store solar cell positions
    #     solar_cells = np.zeros_like(X, dtype=int)  # Use int type for 0 and 1

    #     # Determine the theta range for solar cells (ignoring edges)
    #     theta_start = theta_margin
    #     theta_end = self.n_angular - theta_margin

    #     if cell_thickness == 0:

    #         solar_cells = np.zeros_like(X, dtype=int)  # no solar cells

    #     else:

    #         # Convertir las medidas a número de grids
    #         stripe_width_grids = int(
    #             self.cell_thickness / (self.meshgrid_resolution)
    #         )  # 35 cm -> grids
    #         stripe_spacing_grids = int(
    #             self.cell_spacing / (self.meshgrid_resolution)
    #         )  # 100 cm -> grids
    #         initial_spacing_grids = int(
    #             self.initial_cell_spacing / (self.meshgrid_resolution)
    #         )  # 50 cm -> grids

    #         current_x = initial_spacing_grids
    #         while current_x < self.length_wise_mesh_resolution:
    #             solar_cells[:, current_x : current_x + stripe_width_grids] = 1
    #             current_x += stripe_width_grids + stripe_spacing_grids

    #         # Mark the positions of solar cells based on the parameters
    #         # for i in range(0, int(self.n_angular), int(cell_thickness + cell_spacing)):  # Iterate over columns (Y-direction)
    #         #     solar_cells[int(theta_start):int(theta_end), i:i + int(cell_thickness)] = 1
    #         # plt.figure()
    #         # plt.imshow(solar_cells)
    #         # plt.show()

    #     return (X, Y, Z), solar_cells

    # def generate_distances_grid(self, ground_grid, surface_grid):

    #     X_ground, Y_ground, Z_ground = ground_grid
    #     X_surface, Y_surface, Z_surface = surface_grid

    #     # MODIFICATION
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(projection = '3d')
    #     # surf = ax.plot_surface(X_surface,Y_surface,Z_surface)
    #     # sur2 = ax.plot_surface(X_ground,Y_ground,Z_ground)
    #     # ax.set_ylabel('ylabel')
    #     # ax.set_xlabel('xlabel')
    #     # ax.set_box_aspect([1, 1, 1])  # Keep Scale between x,y,z axes
    #     # plt.show()

    #     distances_list = []
    #     unit_vectors_list = []

    #     # Iterate over each point on the ground grid
    #     for i in range(X_ground.shape[0]):
    #         row_distances = []
    #         row_unit_vectors = []

    #         for j in range(X_ground.shape[1]):
    #             # Extract the ground point
    #             ground_point = np.array(
    #                 [X_ground[i, j], Y_ground[i, j], Z_ground[i, j]]
    #             )

    #             separation_vector_x = X_surface - ground_point[0]
    #             separation_vector_y = Y_surface - ground_point[1]
    #             separation_vector_z = Z_surface - ground_point[2]

    #             # Compute the distance from this ground point to all surface points
    #             distances = np.sqrt(
    #                 (X_surface - ground_point[0]) ** 2
    #                 + (Y_surface - ground_point[1]) ** 2
    #                 + (Z_surface - ground_point[2]) ** 2
    #             )

    #             # Compute the unit vectors by dividing the separation vectors by the distances
    #             unit_vectors_x = separation_vector_x / distances
    #             unit_vectors_y = separation_vector_y / distances
    #             unit_vectors_z = separation_vector_z / distances

    #             # Stack the unit vectors into an array for this ground point
    #             unit_vectors = np.stack(
    #                 (unit_vectors_x, unit_vectors_y, unit_vectors_z), axis=-1
    #             )

    #             # Append the distances and unit vectors to their respective lists
    #             row_distances.append(distances)
    #             row_unit_vectors.append(unit_vectors)

    #         distances_list.append(row_distances)
    #         unit_vectors_list.append(row_unit_vectors)

    #     return distances_list, unit_vectors_list

    # def ground_element_unit_vectors(self):

    #     X = self.generate_ground_grid()[0]
    #     Y = self.generate_ground_grid()[1]
    #     Z = self.generate_ground_grid()[2]

    #     dx = np.gradient(X, axis=0)
    #     dy = np.gradient(Y, axis=0)
    #     dz = np.gradient(Z, axis=0)

    #     # Compute normal vectors using the cross product of the gradients
    #     normals = np.cross(
    #         np.array(
    #             [np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]
    #         ),
    #         np.array([dx, dy, dz]),
    #         axis=0,
    #     )

    #     # Normalize the normal vectors
    #     normals_magnitude = np.linalg.norm(normals, axis=0)
    #     normals_unit = normals / normals_magnitude

    #     return normals_unit, normals_magnitude

    # def surface_element_unit_vectors(self):

    #     X = self.generate_surface()[0][0]
    #     Y = self.generate_surface()[0][1]
    #     Z = self.generate_surface()[0][2]

    #     dx = np.gradient(X, axis=0)
    #     dy = np.gradient(Y, axis=0)
    #     dz = np.gradient(Z, axis=0)

    #     # Compute normal vectors using the cross product of the gradients
    #     normals = np.cross(
    #         np.array(        normals = np.cross(
    #         np.array(
    #             [np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]
    #         ),
    #         np.array([dx, dy, dz]),
    #         axis=0,
    #     )

    #     # Normalize the normal vectors
    #     normals_magnitude = np.linalg.norm(normals, axis=0)
    #     normals_unit = normals / normals_magnitude

    #     return normals_unit, normals_magnitude

    # def surface_tilt(self, normals):

    #     tilt = np.arccos(normals[:][2])

    #     return tilt
