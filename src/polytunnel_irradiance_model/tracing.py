import numpy as np
import pandas as pd
import os

class Tracing:
    def __init__(self, main_tunnel, sun_vecs, surface_grid, surface_tilts, d, R, left_tunnel=None, right_tunnel=None):
        """
        Initialize the Tracing class.

        Parameters:
            main_tunnel (Tunnel): The main polytunnel.
            sun (Sun): The sun object containing sun direction vectors.
            left_tunnel (Tunnel, optional): A neighboring polytunnel to the left.
            right_tunnel (Tunnel, optional): A neighboring polytunnel to the right.
        """
        self.main_tunnel = main_tunnel
        self.left_tunnel = left_tunnel
        self.right_tunnel = right_tunnel
        self.sun_vecs = sun_vecs
        self.surface_grid = surface_grid
        self.surface_tilts = surface_tilts
        self.d = d
        self.R = R

    def ray_intersects_triangle(self, ray_origin, ray_direction, p0, p1, p2):
        """
        Möller–Trumbore intersection algorithm to check if a ray intersects with a triangle.
        
        Parameters:
            ray_origin (np.array): The origin of the ray.
            ray_direction (np.array): The direction of the ray (unit vector).
            p0, p1, p2 (np.array): Vertices of the triangle.
            
        Returns:
            bool: True if the ray intersects the triangle, False otherwise.
        """
        EPSILON = 1e-8
        edge1 = p1 - p0
        edge2 = p2 - p0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        if -EPSILON < a < EPSILON:
            return False  # The ray is parallel to the triangle.

        f = 1.0 / a
        s = ray_origin - p0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return False

        t = f * np.dot(edge2, q)
        if t > EPSILON:  # Ray intersection with the triangle
            return True

        return False  # No valid intersection

    def ray_intersects_surface(self, ray_origin, ray_direction, surface_grid):
        """
        Check if a ray from the given point in the direction of ray_direction intersects
        with any triangle on the surface grid.

        Parameters:
            ray_origin (np.array): The 3D coordinates of the point on the main polytunnel (x, y, z).
            ray_direction (np.array): The sun vector pointing towards the sun (3D unit vector).
            surface_grid (np.array): 3D mesh grid of the polytunnel surface to check against.

        Returns:
            bool: True if the ray intersects the surface, False otherwise.
        """
        # Iterate over all triangles formed by the surface grid
        for i in range(len(surface_grid[0]) - 1):
            for j in range(len(surface_grid[0][0]) - 1):
                # Extract vertices of the first triangle in the grid square
                p0 = np.array([surface_grid[0][i][j], surface_grid[1][i][j], surface_grid[2][i][j]])
                p1 = np.array([surface_grid[0][i+1][j], surface_grid[1][i+1][j], surface_grid[2][i+1][j]])
                p2 = np.array([surface_grid[0][i][j+1], surface_grid[1][i][j+1], surface_grid[2][i][j+1]])

                if self.ray_intersects_triangle(ray_origin, ray_direction, p0, p1, p2):
                    return True

                # Extract vertices of the second triangle in the grid square
                p0 = np.array([surface_grid[0][i+1][j], surface_grid[1][i+1][j], surface_grid[2][i+1][j]])
                p1 = np.array([surface_grid[0][i+1][j+1], surface_grid[1][i+1][j+1], surface_grid[2][i+1][j+1]])
                p2 = np.array([surface_grid[0][i][j+1], surface_grid[1][i][j+1], surface_grid[2][i][j+1]])

                if self.ray_intersects_triangle(ray_origin, ray_direction, p0, p1, p2):
                    return True

        return False
    
    def find_tangent_gradient(self):
        n_rows, n_cols = self.surface_grid[0].shape  # Assuming the grid is n x n

        # Initialize grids to store gradients and angles
        gradients_grid = np.zeros((n_rows, n_cols))
        angles_grid = np.zeros((n_rows, n_cols))
        surface_gradients_grid = np.zeros((n_rows, n_cols))
        surface_angles_grid = np.zeros((n_rows, n_cols))

        x_values = self.surface_grid[0][:, 0]  # x values for the first cross-section
        z_values = self.surface_grid[2][:, 0]  # z values for the first cross-section
        
        # Calculate the gradient for each x, z pair in the first cross-section
        for i in range(len(x_values)):
            x_s = x_values[i]
            z_s = z_values[i]

            if x_s < 0:  # Surface point is to the left of the center
                a = (-2*self.d*x_s-(self.d**2) + (self.R**2) - (x_s**2))
                b = 2*z_s*x_s + 2*z_s*self.d
                c = (self.R**2) - (z_s**2)
                discriminant = (b**2) - 4*a*c

                if discriminant >= 0:
                    if a != 0:
                        gradient = (-b + np.sqrt(discriminant)) / (2*a)
                        surface_gradient = -x_s/z_s

                    else: 
                        gradient = -1e50
                        surface_gradient = 1e50

                    angle_radians = np.arctan(gradient) + np.pi
                    surface_angle_radians = np.arctan(surface_gradient)

            elif x_s > 0:  # Surface point is to the right of the center
                a = (2*self.d*x_s-(self.d**2) + (self.R**2) - (x_s**2))
                b = 2*z_s*x_s - 2*z_s*self.d
                c = (self.R**2) - (z_s**2)
                discriminant = (b**2) - 4*a*c

                if discriminant >= 0:
                    if a != 0:
                        gradient = (-b - np.sqrt(discriminant)) / (2*a)
                        surface_gradient = -x_s/z_s
                    else: 
                        gradient = 1e50
                        surface_gradient = -1e50

                    angle_radians = np.arctan(gradient)
                    surface_angle_radians = np.arctan(surface_gradient) + np.pi

            # Store the calculated gradient and angle for all y-values at this x,z cross-section
            gradients_grid[i, :] = gradient  # Replicate gradient across the row for each y value
            angles_grid[i, :] = angle_radians  # Replicate angle across the row for each y value

            surface_gradients_grid[i, :] = surface_gradient 
            surface_angles_grid[i, :] = surface_angle_radians

        return gradients_grid, angles_grid, surface_gradients_grid, surface_angles_grid
    
    def solid_angle_grid(self, angle_grid1, angle_grid2):
        
        solid_angle_map = np.zeros(angle_grid1.shape)

        for i in range(angle_grid1.shape[0]):
            for j in range(angle_grid1.shape[1]):
                # Define the interval boundaries
                theta1 = angle_grid1[i, j]
                theta2 = angle_grid2[i, j]

                solid_angle_map[i, j] = np.pi*(2- np.cos(theta1) - np.cos(theta2))

        return solid_angle_map
    
    def read_nk_from_csv(self, material):
        """
        Reads wavelength and spectral data from a CSV file for a given material.

        Parameters:
        material (str): The name of the material (used to construct the CSV file name).

        Returns:
        tuple: A tuple containing two NumPy arrays:
            - wavelengths: The wavelengths from the first column.
            - spectral_data: The spectral data from the second column.
        """
        # Construct the file path based on the material name
        file_path = os.path.join('..', 'data', 'materials', f'{material}.csv')
    
        # L# Read CSV file into a DataFrame
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Extract relevant columns
        wavelengths_n_nm = df["λ,n (nm)"].values
        wavelengths_k_nm = df["λ,n (nm)"].values
        n_data = df["n"].values
        k_data = df["k"].values

        return wavelengths_n_nm, wavelengths_k_nm, n_data, k_data
    
    def spectrum_interpolation(self, wavelengths_sample, material):

        wavelengths_n_data, wavelengths_k_data, n_data, k_data = self.read_nk_from_csv(material)

        wavelengths_sample = np.array(wavelengths_sample)
        wavelengths_n_data = np.array(wavelengths_n_data)
        wavelengths_k_data = np.array(wavelengths_k_data)
        n_data = np.array(n_data)
        k_data = np.array(k_data)

        # Perform interpolation
        int_n_data = np.interp(wavelengths_sample, wavelengths_n_data, n_data)
        int_k_data = np.interp(wavelengths_sample, wavelengths_k_data, k_data)

        complex_array = int_n_data + 1j * int_k_data

        return int_n_data, int_k_data, complex_array
    
    def n_list_wavelength(self, mat_list, wavelengths_sample):

        # Generate complex matrices for each material
        complex_mats = [self.spectrum_interpolation(wavelengths_sample, mat)[2] for mat in mat_list] #n.b. material must exclude air
        
        # Create a list of complex arrays at each wavelength, with 1 at the start and end
        complex_array_list = [[1] + [complex_mats[j][i] for j in range(len(mat_list))] + [1] for i in range(len(wavelengths_sample))]

        return complex_array_list
            
        

