import numpy as np

class Polytunnel:

    def __init__(self, radius = 1, length = 10, n_points = 1000, xy_angle = 0, z_angle = 0, x_shift = 0.0, y_shift = 0.0, z_shift = 0.0, theta_margin = 0, cell_thickness = 0, cell_gap = 0):
        self.radius = radius
        self.length = length
        self.n = int(n_points)
        self.xy_angle = xy_angle
        self.z_angle = z_angle
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.theta_margin = theta_margin
        self.cell_thickness = cell_thickness # Thickness of the solar cell (in terms of segments)
        self.cell_gap = cell_gap


    def generate_ground_grid(self):
        y_ground = np.linspace(-self.length, self.length, self.n)
        x_ground = np.linspace(-self.radius+0.0001, self.radius-0.0001, self.n)

        X, Y = np.meshgrid(x_ground, y_ground)
        Z = np.zeros_like(X)

        # Rotation in the XY plane
        cos_xy = np.cos(self.xy_angle)
        sin_xy = np.sin(self.xy_angle)
        
        X_rot_xy = X * cos_xy - Y * sin_xy
        Y = X * sin_xy + Y * cos_xy

        # Rotation around the Z-axis
        cos_z = np.cos(self.z_angle)
        sin_z = np.sin(self.z_angle)

        X = X_rot_xy * cos_z - Z * sin_z
        Z = X_rot_xy * sin_z + Z * cos_z  # Z_final will still be zero since Z is zero initially

        X = X + self.x_shift
        Y = Y + self.y_shift
        Z = Z + self.z_shift


        return (X, Y, Z)
    
    def generate_surface(self):
        # Create the grid in cylindrical coordinates
        y = np.linspace(-self.length, self.length, self.n)
        theta = np.linspace(0, np.pi, self.n)  # Semi-circular cross-section

        # Create a meshgrid for Y and Theta
        Y, Theta = np.meshgrid(y, theta)

        # Convert cylindrical coordinates to Cartesian coordinates
        X = self.radius * np.cos(Theta)  # Horizontal width
        Z = self.radius * np.sin(Theta)  # Height above the ground
        

        # Rotation in the XY plane
        cos_xy = np.cos(self.xy_angle)
        sin_xy = np.sin(self.xy_angle)
        
        X_rot_xy = X * cos_xy - Y * sin_xy
        Y = X * sin_xy + Y * cos_xy

        # Rotation around the Z-axis
        cos_z = np.cos(self.z_angle)
        sin_z = np.sin(self.z_angle)

        X = X_rot_xy * cos_z - Z * sin_z
        Z = X_rot_xy * sin_z + Z * cos_z

        X = X + self.x_shift
        Y = Y + self.y_shift
        Z = Z + self.z_shift

            # Parameters for solar cells
        theta_margin = self.theta_margin  # Margin on the left and right edges (in terms of segments)
        cell_thickness = self.cell_thickness  # Thickness of the solar cell (in terms of segments)
        cell_gap = self.cell_gap  # Gap between solar cells (in terms of segments)

            # Initialize an array to store solar cell positions
        solar_cells = np.zeros_like(X, dtype=int)  # Use int type for 0 and 1

        # Determine the theta range for solar cells (ignoring edges)
        theta_start = theta_margin
        theta_end = self.n - theta_margin
        
        if cell_thickness == 0:

            solar_cells = np.zeros_like(X, dtype=int)  #no solar cells
        
        else:
            # Mark the positions of solar cells based on the parameters
            for i in range(0, self.n, cell_thickness + cell_gap):  # Iterate over columns (Y-direction)
                solar_cells[theta_start:theta_end, i:i + cell_thickness] = 1 

        return (X, Y, Z), solar_cells
    
    def generate_distances_grid(self, ground_grid, surface_grid):
        
        X_ground, Y_ground, Z_ground = ground_grid
        X_surface, Y_surface, Z_surface = surface_grid
        
        distances_list = []
        unit_vectors_list = []

        # Iterate over each point on the ground grid
        for i in range(X_ground.shape[0]):
            row_distances = []
            row_unit_vectors = []

            for j in range(X_ground.shape[1]):
                # Extract the ground point
                ground_point = np.array([X_ground[i, j], Y_ground[i, j], Z_ground[i, j]])
                
                separation_vector_x = X_surface - ground_point[0]
                separation_vector_y = Y_surface - ground_point[1]
                separation_vector_z = Z_surface - ground_point[2]

                # Compute the distance from this ground point to all surface points
                distances = np.sqrt((X_surface - ground_point[0])**2 + 
                                    (Y_surface - ground_point[1])**2 + 
                                    (Z_surface - ground_point[2])**2)
                
                # Compute the unit vectors by dividing the separation vectors by the distances
                unit_vectors_x = separation_vector_x / distances
                unit_vectors_y = separation_vector_y / distances
                unit_vectors_z = separation_vector_z / distances
                
                # Stack the unit vectors into an array for this ground point
                unit_vectors = np.stack((unit_vectors_x, unit_vectors_y, unit_vectors_z), axis=-1)
                
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
        normals = np.cross(np.array([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]), np.array([dx, dy, dz]), axis=0)

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
        normals = np.cross(np.array([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)]), np.array([dx, dy, dz]), axis=0)

        # Normalize the normal vectors
        normals_magnitude = np.linalg.norm(normals, axis=0)
        normals_unit = normals / normals_magnitude

        return normals_unit, normals_magnitude
    
    
    def surface_tilt(self, normals):

        tilt = np.arccos(normals[:][2])

        return tilt
    

