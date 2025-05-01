import numpy as np
class Angle:

    def __init__(self, main_tunnel, surface_grid, d, R):
        """
        Initialize the Tracing class.

        Parameters:
            main_tunnel (Tunnel): The main polytunnel.
            sun (Sun): The sun object containing sun direction vectors.
            left_tunnel (Tunnel, optional): A neighboring polytunnel to the left.
            right_tunnel (Tunnel, optional): A neighboring polytunnel to the right.
        """
        self.main_tunnel = main_tunnel
        self.surface_grid = surface_grid
        self.d = d
        self.radius = R
    # Compute visible solid angle function
    
    def calculate_max_tangent_angles_grid(self):
    # Extract grid dimensions
        n_rows, n_cols = self.surface_grid[0].shape

        # Initialize the grid for maximum angles
        angles_grid = np.zeros((n_rows, n_cols))

        # Extract the unique x coordinates (assuming they are along the rows)
        x_values = self.surface_grid[0][:, 0]
        z_values = self.surface_grid[2][0, :]  # z-values along one row (same for all rows)

        # Iterate over unique x coordinates (rows of the grid)
        for i, x_s in enumerate(x_values):
            max_angles_row = []

            # Determine which polytunnel to consider
            for z_s in z_values:
                if x_s < 0:  # Point is to the left of the center
                    max_angle_left = 0

                    # Calculate max angle for the left polytunnel
                    for z_t in np.linspace(-self.R, self.R, 1000):
                        x_t_left = -self.d + np.sqrt(self.R**2 - z_t**2)
                        angle_left = np.arctan2(z_s - z_t, x_s - x_t_left)
                        max_angle_left = max(max_angle_left, angle_left)

                    max_angles_row.append(max_angle_left)

                elif x_s > 0:  # Point is to the right of the center
                    max_angle_right = 0

                    # Calculate max angle for the right polytunnel
                    for z_t in np.linspace(-self.R, self.R, 1000):
                        x_t_right = self.d - np.sqrt(self.R**2 - z_t**2)
                        angle_right = np.arctan2(z_s - z_t, x_s - x_t_right)
                        max_angle_right = max(max_angle_right, angle_right)

                    max_angles_row.append(max_angle_right)

                else:  # Point is exactly at the center (x_s = 0)
                    max_angle = np.pi
                    max_angles_row.append(max_angle)  # No tangent can be calculated exactly at the center

            # Fill the entire row with computed angles
            angles_grid[i, :] = max_angles_row

        return angles_grid