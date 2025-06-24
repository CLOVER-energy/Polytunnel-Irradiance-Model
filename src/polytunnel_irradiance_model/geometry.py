import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Polytunnel:

    def __init__(
        self,
        radius1=1,
        length=10,
        tilt=0,
        azimuthal_orientation=0,
        x_shift=0.0,
        y_shift=0.0,
        z_shift=0.0,
        theta_margin=0,
        cell_thickness=0,
        cell_gap=0,
        radius2=None,
        res_meshgrid=1.0,
        initial_cell_gap=0.0,
    ):
        self.res_meshgrid = res_meshgrid
        self.radius1 = radius1
        self.length = length
        self.n = int(self.length / self.res_meshgrid)
        self.n_angular = int(self.length / self.res_meshgrid)
        self.tilt = tilt
        self.azimuthal_orientation = azimuthal_orientation
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.theta_margin = theta_margin
        self.cell_thickness = (
            cell_thickness  # Thickness of the solar cell (in terms of segments)
        )
        self.cell_gap = cell_gap
        self.initial_cell_gap = initial_cell_gap

        if radius2:
            # If radius2 in arguments, this will be the secondary radius, to obtain an elliptical shape
            self.radius2 = radius2
        else:
            # Otherwise it will continue being a circular shape
            self.radius2 = radius1

    # MODIFICATION
    def apply_rotations(self, X, Y, Z):
        """
        Appy two rotations:
        - First: Around Z by azimuthal_orientation (in radians).
        - Second: Around X by tilt  (in radians).
        """
        # Rotation around Z
        R_z = np.array(
            [
                [
                    np.cos(self.azimuthal_orientation),
                    -np.sin(self.azimuthal_orientation),
                    0,
                ],
                [
                    np.sin(self.azimuthal_orientation),
                    np.cos(self.azimuthal_orientation),
                    0,
                ],
                [0, 0, 1],
            ]
        )

        # Rotation around X
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.tilt), -np.sin(self.tilt)],
                [0, np.sin(self.tilt), np.cos(self.tilt)],
            ]
        )

        # Flat coordinates to make an easier rotation
        coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

        # Apply rotations sequentially
        rotated_coords = R_z @ (R_x @ coords)

        # Reshape coordinates to keep same as input X,Y,Z arrays
        X_rot, Y_rot, Z_rot = rotated_coords.reshape(3, *X.shape)
        return X_rot, Y_rot, Z_rot

    def generate_ground_grid(self):
        # y_ground = np.linspace(-self.length, self.length, self.n)
        y_ground = np.linspace(0, self.length, self.n)
        x_ground = np.linspace(
            -self.radius1 + 0.0001, self.radius1 - 0.0001, self.n_angular
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
               radius1 * cos(theta),
               length,
               radius2 * sin(theta)
           ],
        """

        # Create the grid in cylindrical coordinates
        # y = np.linspace(-self.length, self.length, self.n)
        y = np.linspace(0, self.length, self.n)
        theta = np.linspace(0, np.pi, self.n_angular)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.radius1 * np.cos(Theta)  # Horizontal width
        Z = self.radius2 * np.sin(Theta)  # Height above the ground

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
        cell_gap = self.cell_gap  # Gap between solar cells (in terms of segments)

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
                self.cell_thickness / (self.res_meshgrid)
            )  # 35 cm -> grids
            stripe_spacing_grids = int(
                self.cell_gap / (self.res_meshgrid)
            )  # 100 cm -> grids
            initial_spacing_grids = int(
                self.initial_cell_gap / (self.res_meshgrid)
            )  # 50 cm -> grids

            current_x = initial_spacing_grids
            while current_x < self.n:
                solar_cells[:, current_x : current_x + stripe_width_grids] = 1
                current_x += stripe_width_grids + stripe_spacing_grids

            # Mark the positions of solar cells based on the parameters
            # for i in range(0, int(self.n_angular), int(cell_thickness + cell_gap)):  # Iterate over columns (Y-direction)
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
