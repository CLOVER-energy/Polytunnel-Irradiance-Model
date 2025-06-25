import enum
import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import acos, asin, cos, degrees, isnan, radians, pi, sin
from typing import TypeVar

import numpy as np


# Floating point precision:
#   The floating-point precision of the numbers to use when doing rotations.
FLOATING_POINT_PRECISION: int = 8


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
    _azimuth_rotation_matrix: list[list[float]] | None = None
    _tilt_rotation_matrix: list[list[float]] | None = None

    def __init_subclass__(cls, curve_type: CurveType) -> None:
        """
        Hook used to store the type of the curve.

        :param: **curve_type:**
            The type of the curve.

        """

        cls.curve_type = curve_type  # type: ignore [attr-defined]
        TYPE_TO_CURVE_MAPPING[curve_type] = cls

        super().__init_subclass__()

    @abstractmethod
    def get_angles_from_surface_displacement(
        self, displacement: float
    ) -> tuple[float, float]:
        """
        Abstract method that must be implemented in subclasses.
        Calculate the azimuth and zenith angles at a point along the curve.

        :param: **displacement:**
            The distance from the central axis.

        :returns:
            - A tuple, (azimuth, tilt), with angles in degrees.

        """

        raise NotImplementedError("This method must be implemented in subclasses")

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
            self._azimuth_rotation_matrix = [
                [
                    cos(radians(self.zenith_rotation_angle)),
                    -sin(radians(self.zenith_rotation_angle)),
                    0,
                ],
                [
                    sin(radians(self.zenith_rotation_angle)),
                    cos(radians(self.zenith_rotation_angle)),
                    0,
                ],
                [0, 0, 1],
            ]

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
            self._tilt_rotation_matrix = [
                [1, 0, 0],
                [
                    0,
                    cos(radians(self.tilt_rotation_angle)),
                    -sin(radians(self.tilt_rotation_angle)),
                ],
                [
                    0,
                    sin(radians(self.tilt_rotation_angle)),
                    cos(radians(self.tilt_rotation_angle)),
                ],
            ]

        return self._tilt_rotation_matrix

    def _get_rotated_angles_from_surface_normal(
        self, un_rotated_normal: list[float]
    ) -> tuple[float, float]:
        """
        Rotate the normal vector based on the orientation of the curve/Polytunnel.

        :param: **un_rotated_normal:**
            The surface normal vector in the un-rotated frame.

        :returns:
            - A `tuple` containing information about the new rotated normal vector:
                - The azimuth angle, in degrees,
                - THe tilt angle, in degrees.

        """

        # Rotate this normal vector based on the tilt and azimuth of the polytunnel.
        rotated_normal = np.matmul(
            self.azimuth_rotation_matrix,
            np.matmul(self.tilt_rotation_matrix, un_rotated_normal),
        )

        # Compute the new azimuth and tilt angles based on these rotations.
        # The tilt angle is simply the z component of the vector.
        tilt_angle = round(acos(rotated_normal[2]), FLOATING_POINT_PRECISION)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                x_y_plane_component = sin(tilt_angle)
            except (RuntimeError, RuntimeWarning, ZeroDivisionError):
                azimuth_angle = pi
            else:
                if round(rotated_normal[1], 6) == 0:
                    azimuth_angle = (
                        pi
                        if rotated_normal[0] == 0
                        else pi / 2 if (rotated_normal[0] > 0) else -pi / 2
                    )
                else:
                    arccos_angle = acos(
                        round(
                            rotated_normal[1] / x_y_plane_component,
                            FLOATING_POINT_PRECISION,
                        )
                    )
                    azimuth_angle = (
                        2 * pi - arccos_angle if rotated_normal[0] < 0 else arccos_angle
                    )

        # Check that the tilt is not out-of-bounds
        if tilt_angle > pi / 2:
            raise UndergroundCellError(
                f"A cell in the module has a tilt angle of {tilt_angle} radians, which "
                "is underground."
            )

        # Check that the azimuth angle is not `nan` and set it to South if so.
        if isnan(azimuth_angle):
            azimuth_angle = pi

        # Return these angles in degrees
        return degrees(azimuth_angle) % 360, degrees(tilt_angle)


@dataclass(kw_only=True)
class CircularCurve(Curve, curve_type=CurveType.CIRCULAR):
    """
    Represents a circular geometry. In this instance, a radius of curvature is required.

    .. attribute:: radius_of_curvature:
        Represents the radius of curvature.

    """

    radius_of_curvature: float

    def get_angles_from_surface_displacement(
        self, displacement: float
    ) -> tuple[float, float]:
        """
        Calculate the azimuth and zenith angles at a point along the curve.

        :param: **displacement:**
            The distance from the central axis.

        :returns:
            - A tuple, (azimuth, tilt), with angles in degrees.

        """

        # Compute the zenith angle in radians based on the distance from the axis.
        zenith_angle: float = displacement / self.radius_of_curvature

        # Don't both calculating if the displacement is zero
        if displacement == 0:
            return (180, self.axis_tilt)

        # Compute the components of a unit normal vector with this zenith angle.
        un_rotated_normal: list[float] = [sin(zenith_angle), 0, cos(zenith_angle)]

        return self._get_rotated_angles_from_surface_normal(un_rotated_normal)


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

    def get_angles_from_surface_displacement(
        self, displacement: float
    ) -> tuple[float, float]:
        """
        Calculate the azimuth and zenith angles at a point along the curve.

        :param: **displacement:**
            The distance from the central axis.

        :returns:
            - A tuple, (azimuth, tilt), with angles in degrees.

        """

        # Compute the zenith angle in radians based on the distance from the axis.
        zenith_angle: float = displacement / self.radius_of_curvature

        # Don't both calculating if the displacement is zero
        if displacement == 0:
            return (180, self.axis_tilt)

        # Compute the components of a unit normal vector with this zenith angle.
        un_rotated_normal: list[float] = [sin(zenith_angle), 0, cos(zenith_angle)]

        return self._get_rotated_angles_from_surface_normal(un_rotated_normal)


class ModuleType(enum.Enum):
    """
    Denotes the type of the module.

    - CRYSTALINE:
        Used to distinguish crystaline Silicon modules.

    - THIN_FILM:
        Used to distinguish thin-film modules.

    """

    CRYSTALINE: str = "c_Si"
    THIN_FILM: str = "thin_film"


@dataclass
class PVCellMaterial:
    """
    Represents a layer within the PV module.

    .. attribute:: thickness:
        The thickness of the layer.

    """

    thickness: float


@dataclass
class PVModule:
    """
    Contains information about the PV module.

    .. attribute:: length:
        The length of the module, in meters.

    .. attribute:: materials:
        The materials included, along with their thicknesses.

    .. attribute:: multistack:
        The number of stacks to include

    .. attribute:: orientation:
        The angle between the axis of the polytunnel and of the module.

    .. attribute:: width:
        The width of the cell, measured in metres.

    """

    length: float
    materials: list[PVCellMaterial]
    module_type: ModuleType
    multistack: int | None
    orientation: float
    width: float


class Polytunnel:
    """
    Represents a Polytunnel instance.

    .. attribute:: curve:
        The curve which defines the shape of the polytunnel.

    .. attribute:: length:
        The length of the polytunnel.

    .. attribute:: meshgrid_resolution:
        The resolution of meshgrid, in meters.

    .. attribute:: pv_module:
        The pv module on the polytunnel.

    .. attribute:: width:
        The width of the polytunnel at the point being considered, in meters.

    """

    def __init__(
        self,
        curve: Curve,
        length: float,
        meshgrid_resolution: float,
        pv_module: PVModule,
        width: float,
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
        self.pv_module = pv_module
        self.width = width

        self.length_wise_mesh_resolution = int(self.length / self.meshgrid_resolution)
        # FIXME: self.n_angular = int(self.width / self.meshgrid_resolution)

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

    def generate_ground_grid(self):
        # y_ground = np.linspace(-self.length, self.length, self.n)
        y_ground = np.linspace(0, self.length, self.length_wise_mesh_resolution)
        x_ground = np.linspace(
            -self.semi_major_axis + 0.0001,
            self.semi_major_axis - 0.0001,
            self.n_angular,
        )

        X, Y = np.meshgrid(x_ground, y_ground)
        Z = np.zeros_like(X)

        # MODIFICATION
        # Rotation in the XY plane
        # cos_xy = np.cos(self.tilt)
        # sin_xy = np.sin(self.tilt)

        # X_rot_xy = X * cos_xy - Y * sin_xy
        # Y = X * sin_xy + Y * cos_xy

        # # Rotation around the Z-axis
        # cos_z = np.cos(self.azimuthal_orientation)
        # sin_z = np.sin(self.azimuthal_orientation)

        # X = X_rot_xy * cos_z - Z * sin_z
        # Z = X_rot_xy * sin_z + Z * cos_z  # Z_final will still be zero since Z is zero initially

        if self.tilt != 0 or self.azimuthal_orientation != 0:
            X, Y, Z = self.apply_rotations(X, Y, Z)

        X = X + self.x_shift
        Y = Y + self.y_shift
        Z = Z + self.z_shift

        return (X, Y, Z)

    def generate_surface(self):
        """
        The parametric definition of this cylinder surface (S) is:
           S = [
               semi_major_axis * cos(theta),
               length,
               semi_minor_axis * sin(theta)
           ],
        """

        # Create the grid in cylindrical coordinates
        # y = np.linspace(-self.length, self.length, self.n)
        y = np.linspace(0, self.length, self.length_wise_mesh_resolution)
        theta = np.linspace(0, np.pi, self.n_angular)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.semi_major_axis * np.cos(Theta)  # Horizontal width
        Z = self.semi_minor_axis * np.sin(Theta)  # Height above the ground

        # MODIFICATION
        # # Rotation in the XY plane
        # cos_xy = np.cos(self.tilt)
        # sin_xy = np.sin(self.tilt)

        # X_rot_xy = X * cos_xy - Y * sin_xy
        # Y = X * sin_xy + Y * cos_xy

        # # Rotation around the Z-axis
        # cos_z = np.cos(self.azimuthal_orientation)
        # sin_z = np.sin(self.azimuthal_orientation)

        # X = X_rot_xy * cos_z - Z * sin_z
        # Z = X_rot_xy * sin_z + Z * cos_z
        if self.tilt != 0 or self.azimuthal_orientation != 0:
            X, Y, Z = self.apply_rotations(X, Y, Z)

        X = X + self.x_shift
        Y = Y + self.y_shift
        Z = Z + self.z_shift

        # Parameters for solar cells
        theta_margin = (
            self.theta_margin
        )  # Margin on the left and right edges (in terms of segments)
        cell_thickness = (
            self.cell_thickness
        )  # Thickness of the solar cell (in terms of segments)
        cell_spacing = (
            self.cell_spacing
        )  # Gap between solar cells (in terms of segments)

        # Initialize an array to store solar cell positions
        solar_cells = np.zeros_like(X, dtype=int)  # Use int type for 0 and 1

        # Determine the theta range for solar cells (ignoring edges)
        theta_start = theta_margin
        theta_end = self.n_angular - theta_margin

        if cell_thickness == 0:

            solar_cells = np.zeros_like(X, dtype=int)  # no solar cells

        else:

            # Convertir las medidas a número de grids
            stripe_width_grids = int(
                self.cell_thickness / (self.meshgrid_resolution)
            )  # 35 cm -> grids
            stripe_spacing_grids = int(
                self.cell_spacing / (self.meshgrid_resolution)
            )  # 100 cm -> grids
            initial_spacing_grids = int(
                self.initial_cell_spacing / (self.meshgrid_resolution)
            )  # 50 cm -> grids

            current_x = initial_spacing_grids
            while current_x < self.length_wise_mesh_resolution:
                solar_cells[:, current_x : current_x + stripe_width_grids] = 1
                current_x += stripe_width_grids + stripe_spacing_grids

            # Mark the positions of solar cells based on the parameters
            # for i in range(0, int(self.n_angular), int(cell_thickness + cell_spacing)):  # Iterate over columns (Y-direction)
            #     solar_cells[int(theta_start):int(theta_end), i:i + int(cell_thickness)] = 1
            # plt.figure()
            # plt.imshow(solar_cells)
            # plt.show()

        return (X, Y, Z), solar_cells

    def generate_distances_grid(self, ground_grid, surface_grid):

        X_ground, Y_ground, Z_ground = ground_grid
        X_surface, Y_surface, Z_surface = surface_grid

        # MODIFICATION
        # fig = plt.figure()
        # ax = fig.add_subplot(projection = '3d')
        # surf = ax.plot_surface(X_surface,Y_surface,Z_surface)
        # sur2 = ax.plot_surface(X_ground,Y_ground,Z_ground)
        # ax.set_ylabel('ylabel')
        # ax.set_xlabel('xlabel')
        # ax.set_box_aspect([1, 1, 1])  # Keep Scale between x,y,z axes
        # plt.show()

        distances_list = []
        unit_vectors_list = []

        # Iterate over each point on the ground grid
        for i in range(X_ground.shape[0]):
            row_distances = []
            row_unit_vectors = []

            for j in range(X_ground.shape[1]):
                # Extract the ground point
                ground_point = np.array(
                    [X_ground[i, j], Y_ground[i, j], Z_ground[i, j]]
                )

                separation_vector_x = X_surface - ground_point[0]
                separation_vector_y = Y_surface - ground_point[1]
                separation_vector_z = Z_surface - ground_point[2]

                # Compute the distance from this ground point to all surface points
                distances = np.sqrt(
                    (X_surface - ground_point[0]) ** 2
                    + (Y_surface - ground_point[1]) ** 2
                    + (Z_surface - ground_point[2]) ** 2
                )

                # Compute the unit vectors by dividing the separation vectors by the distances
                unit_vectors_x = separation_vector_x / distances
                unit_vectors_y = separation_vector_y / distances
                unit_vectors_z = separation_vector_z / distances

                # Stack the unit vectors into an array for this ground point
                unit_vectors = np.stack(
                    (unit_vectors_x, unit_vectors_y, unit_vectors_z), axis=-1
                )

                # Append the distances and unit vectors to their respective lists
                row_distances.append(distances)
                row_unit_vectors.append(unit_vectors)

            distances_list.append(row_distances)
            unit_vectors_list.append(row_unit_vectors)

        return distances_list, unit_vectors_list

    def ground_element_unit_vectors(self):

        X = self.generate_ground_grid()[0]
        Y = self.generate_ground_grid()[1]
        Z = self.generate_ground_grid()[2]

        dx = np.gradient(X, axis=0)
        dy = np.gradient(Y, axis=0)
        dz = np.gradient(Z, axis=0)

        # Compute normal vectors using the cross product of the gradients
        normals = np.cross(
            np.array(
                [np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]
            ),
            np.array([dx, dy, dz]),
            axis=0,
        )

        # Normalize the normal vectors
        normals_magnitude = np.linalg.norm(normals, axis=0)
        normals_unit = normals / normals_magnitude

        return normals_unit, normals_magnitude

    def surface_element_unit_vectors(self):

        X = self.generate_surface()[0][0]
        Y = self.generate_surface()[0][1]
        Z = self.generate_surface()[0][2]

        dx = np.gradient(X, axis=0)
        dy = np.gradient(Y, axis=0)
        dz = np.gradient(Z, axis=0)

        # Compute normal vectors using the cross product of the gradients
        normals = np.cross(
            np.array(
                [np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]
            ),
            np.array([dx, dy, dz]),
            axis=0,
        )

        # Normalize the normal vectors
        normals_magnitude = np.linalg.norm(normals, axis=0)
        normals_unit = normals / normals_magnitude

        return normals_unit, normals_magnitude

    def surface_tilt(self, normals):

        tilt = np.arccos(normals[:][2])

        return tilt
