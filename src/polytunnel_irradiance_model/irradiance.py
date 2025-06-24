import numpy as np
import matplotlib.pyplot as plt
import tmm_fast as tmm
from scipy.integrate import trapezoid
import tracing as tracer
import os
import pandas as pd
import torch


class TunnelIrradiance:
    def __init__(self, polytunnel, radius1, radius2, length):
        """
        Initializes the RayTracer class for calculating solar irradiance on the polytunnel ground.

        :param polytunnel: Instance of the Polytunnel class.
        """
        self.polytunnel = polytunnel
        self.radius1 = radius1
        self.radius2 = radius2
        self.length = length

    def irradiance_rays(self, normals_unit, sun_positions, sun_vecs, irradiance_frames):

        irradiance = []

        for i in range(len(sun_positions)):
            sun_vec = sun_vecs[i]
            calc = self.calculate_irradiance(
                normals_unit, sun_vec, irradiance_frames[i]
            )
            irradiance.append(calc)
        return np.array(irradiance)

    def calculate_irradiance(self, normals_unit, sun_vec, spectral_irradiance):

        irradiance_surface = np.clip(
            (spectral_irradiance * np.tensordot(normals_unit, sun_vec, axes=(0, 0))),
            a_min=0,
            a_max=None,
        )

        return irradiance_surface

    def shading_exposure_map(self, angle_grid1, angle_grid2, sun_vecs):

        exposure_maps = []
        for i in range(len(sun_vecs)):
            x = sun_vecs[i][0]
            z = sun_vecs[i][2]
            theta = np.arctan2(z, x)
            # Initialize exposure map with zeros
            exposure_map = np.zeros(angle_grid1.shape, dtype=int)

            # Iterate over each cell in the grids
            for i in range(angle_grid1.shape[0]):
                for j in range(angle_grid1.shape[1]):
                    # Define the interval boundaries
                    theta1 = angle_grid1[i, j]
                    theta2 = angle_grid2[i, j]

                    if theta1 > theta2:
                        # theta1 is obtuse
                        if theta2 <= theta <= theta1:
                            exposure_map[i, j] = 1

                    elif theta2 > theta1:
                        # theta2 is obtuse
                        if theta1 <= theta <= theta2:
                            exposure_map[i, j] = 1

            exposure_maps.append(exposure_map)

        return exposure_maps

    def diffuse_irradiance_ground(
        self,
        distances_grid,
        separation_unit_vector_grid,
        normals_unit_ground,
        normals_unit_surface,
        surface_areas,
        irradiance_frames,
        transmissivity,
    ):

        diffuse_irradiance_frames = []

        ground_shape = (len(distances_grid), len(distances_grid[0]))

        for j in range(len(irradiance_frames)):
            # irradiance_ground = np.zeros(ground_shape)
            irradiance_ground = np.zeros(ground_shape, dtype=object)

            # Iterate over each point on the ground grid
            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):

                    total_irradiance = 0.0

                    # Iterate over each point on the surface grid
                    surface_shape = irradiance_frames[j].shape
                    # for k in range(surface_shape[0]):
                    #     for l in range(surface_shape[1]):
                    for k in range(surface_shape[1]):
                        for l in range(surface_shape[0]):

                            # Sum the irradiance contribution from each surface point
                            distance = distances_grid[p][q][k][l]
                            ground_projection = np.dot(
                                normals_unit_ground[:, p, q],
                                separation_unit_vector_grid[p][q][k][l],
                            )
                            surface_projection = np.dot(
                                normals_unit_surface[:, k, l],
                                separation_unit_vector_grid[p][q][k][l],
                            )

                            # irradiance = (1-transmissivity) * ground_projection * surface_projection * surface_areas[k][l] * irradiance_frames[j][k][l] / (2*np.pi*(distance)**2)
                            irradiance = (
                                (1 - transmissivity)
                                * ground_projection
                                * surface_projection
                                * surface_areas[k][l]
                                * irradiance_frames[j][l][k]
                                / (2 * np.pi * (distance) ** 2)
                            )
                            total_irradiance += irradiance

                    irradiance_ground[p][q] = total_irradiance

            diffuse_irradiance_frames.append(irradiance_ground)

        return diffuse_irradiance_frames

    def direct_irradiance_ground(
        self, normals_unit_ground, sun_vecs, irradiance_frames, transmissivity
    ):

        direct_irradiance_frames = []

        for i in range(len(irradiance_frames)):

            sun_vec = sun_vecs[i]

            # The next line returns some errors handling the clipping of the calculation
            # calc = np.clip((transmissivity * irradiance_frames[i] * np.tensordot(normals_unit_ground, sun_vec, axes=(0, 0))), a_min = 0, a_max = None)

            # We perfom the same calculation in different steps
            dot_product = np.tensordot(
                normals_unit_ground, sun_vec, axes=(0, 0)
            )  # Shape (n_points, n_points_angular)
            dot_product = np.squeeze(dot_product)  # Delete additional dimensions

            calc = (
                transmissivity * irradiance_frames[i] * dot_product
            )  # Perform calculation
            calc_clipped = np.zeros_like(calc)

            # we perform clipping by using ReLU Function
            # calc = np.clip(array,  a_min = 0, a_max = None) is equivalent to relu(array)
            # since negative values are clipped to 0 and there is not an upper limit

            relu = torch.nn.ReLU()
            for first_index_shape, array_item in enumerate(calc):
                for second_index_shape, inside_array in enumerate(array_item):
                    # We perform relu for every individual array within calc
                    # and clip using relu
                    clipped_val = relu(torch.from_numpy(inside_array))
                    calc_clipped[first_index_shape][
                        second_index_shape
                    ] = clipped_val.numpy()

            # direct_irradiance_frames.append(calc)
            direct_irradiance_frames.append(calc_clipped)

        return direct_irradiance_frames

    def ray_trace_to_surface(
        self,
        ground_grid,
        normals_unit_ground,
        surface_grid,
        distances_grid,
        sun_vecs,
        irradiance_frames,
        transmissivity,
    ):

        ground_shape = (len(distances_grid), len(distances_grid[0]))
        irradiance_traced = []

        for i in range(len(irradiance_frames)):
            irradiance_grid = np.zeros((ground_shape[0], ground_shape[1]))

            # Calculate intersections and closest points
            for j in range(ground_shape[0]):
                for k in range(ground_shape[1]):
                    # Get ground point
                    x_g = ground_grid[0][j][k]
                    y_g = ground_grid[1][j][k]
                    z_g = ground_grid[2][j][k]

                    # Normalize sun vector
                    sun_vec = np.array(sun_vecs[i])
                    sun_vec = sun_vec / np.linalg.norm(sun_vec)

                    # Compute t for intersection with the cylindrical surface
                    # a = sun_vec[0]**2 + sun_vec[2]**2
                    # b = 2 * (x_g * sun_vec[0] + z_g * sun_vec[2])
                    # c = x_g**2 + z_g**2 - self.radius**2
                    # Compute t for intersection with the cylindrical surface (elliptical shape)
                    a = (sun_vec[0] / self.radius1) ** 2 + (
                        sun_vec[2] / self.radius2
                    ) ** 2
                    b = 2 * (
                        x_g * sun_vec[0] / (self.radius1) ** 2
                        + z_g * sun_vec[2] / (self.radius2) ** 2
                    )
                    c = x_g**2 / (self.radius1) ** 2 + z_g**2 / (self.radius2) ** 2 - 1

                    discriminant = b**2 - 4 * a * c

                    if discriminant >= 0:
                        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

                        # Choose the correct t (positive and smallest)
                        t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)

                        # Calculate intersection point on the cylindrical surface
                        x_int = x_g + t * sun_vec[0]
                        y_int = y_g + t * sun_vec[1]
                        z_int = z_g + t * sun_vec[2]

                        # Check if the intersection is within the tunnel bounds
                        if 0 <= y_int <= self.length:
                            # Find the closest surface grid point
                            distances = np.sqrt(
                                (surface_grid[0] - x_int) ** 2
                                + (surface_grid[1] - y_int) ** 2
                                + (surface_grid[2] - z_int) ** 2
                            )
                            closest_j, closest_k = np.unravel_index(
                                np.argmin(distances), distances.shape
                            )
                            irradiance_point = irradiance_frames[i][closest_j][
                                closest_k
                            ]
                        else:
                            irradiance_point = 0

                    else:
                        irradiance_point = 0

                    # Apply transmissivity and set the irradiance in the grid
                    irradiance_grid[j][k] = (
                        transmissivity
                        * irradiance_point
                        * np.dot(sun_vecs[i], normals_unit_ground[:, j, k])
                    )

            irradiance_traced.append(irradiance_grid)

        return irradiance_traced

    def global_irradiance_ground(
        self, direct_irradiance_ground, diffuse_irradiance_ground
    ):

        global_irradiance_frames = []

        for i in range(len(direct_irradiance_ground)):

            calc = direct_irradiance_ground[i] + diffuse_irradiance_ground[i]

            global_irradiance_frames.append(calc)

        return global_irradiance_frames

    def power(self, area_grid, irradiance_frames):

        power_total = []

        ground_shape = (len(area_grid), len(area_grid[0]))

        for j in range(len(irradiance_frames)):
            power_ground = np.zeros(ground_shape, dtype="object")
            total_power = 0
            # Iterate over each point on the ground grid
            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):

                    # Store the result in the irradiance_ground array
                    power_calc = irradiance_frames[j][p][q] * area_grid[p][q]
                    power_ground[p][q] = power_calc

                    total_power += power_calc

            power_total.append(total_power)

        return power_total

    def shaded_irradiance_rays(self, irradiance_frames, shaded_exposure_maps):

        shaded_irradiance_frames = []

        for i in range(len(irradiance_frames)):
            new_map = np.where(shaded_exposure_maps[i] == 1, irradiance_frames[i], 0)
            shaded_irradiance_frames.append(new_map)

        return shaded_irradiance_frames

    def shaded_irradiance_spectra(
        self, irradiance_frames_spectra, shaded_exposure_maps
    ):

        shaded_irradiance_frames = np.zeros_like(irradiance_frames_spectra)
        # Iterate over the indices of irradiance_frames_spectra and shaded_exposure_maps
        for i in range(len(irradiance_frames_spectra)):
            for j in range(len(irradiance_frames_spectra[i])):
                for k in range(len(irradiance_frames_spectra[i][j])):

                    if shaded_exposure_maps[i][j][k] == 1:
                        # If exposure map is 1, copy the entire array at irradiance_frames_spectra[i][j][k]
                        shaded_irradiance_frames[i][j][k] = irradiance_frames_spectra[
                            i
                        ][j][k]

                    else:
                        # If exposure map is 0, store an array of zeros with the same shape as irradiance_frames_spectra[i][j][k]
                        shaded_irradiance_frames[i][j][k] = np.zeros_like(
                            irradiance_frames_spectra[i][j][k]
                        )

        return shaded_irradiance_frames

    def transmitted_absorbed_spectra(self, wavelengths_sample, spectra, scalar_grid):

        # L# Read CSV file into a DataFrame
        file_path = os.path.join("..", "data", "polyethene-transmission.csv")
        df = pd.read_csv(file_path, skipinitialspace=True)

        # Extract relevant columns
        wavelengths_n_nm = df["λ,n (nm)"].values
        t = df["t"].values

        wavelengths_sample = np.array(wavelengths_sample)
        wavelengths_n_nm = np.array(wavelengths_n_nm)
        t = np.array(t)
        a = 1 - t

        # Perform interpolation
        int_t_data = np.interp(wavelengths_sample, wavelengths_n_nm, t)
        int_a_data = np.interp(wavelengths_sample, wavelengths_n_nm, a)

        transmitted_spectra = np.zeros_like(spectra)
        absorbed_spectra = np.zeros_like(spectra)

        # Iterate over the indices of irradiance_frames_spectra and shaded_exposure_maps
        for i in range(len(spectra)):
            for j in range(len(spectra[i])):
                for k in range(len(spectra[i][j])):
                    # If exposure map is 1, copy the entire array at irradiance_frames_spectra[i][j][k]
                    transmitted_spectra[i][j][k] = (
                        spectra[i][j][k] * int_t_data * scalar_grid[i][j][k]
                    )
                    absorbed_spectra[i][j][k] = (
                        spectra[i][j][k] * int_a_data * scalar_grid[i][j][k]
                    )

        return transmitted_spectra, absorbed_spectra

    def eg_t_grid(self, material_list, d_list):

        optical_wavelengths = np.linspace(300, 700, 1000)
        Theta = np.linspace(-np.pi / 2, np.pi / 2, 1000)  # Incident angles (Theta)
        complex_array = tracer.n_list_wavelength(material_list, optical_wavelengths)
        optical_wavelengths = np.array(
            optical_wavelengths, dtype=np.float64
        )  # Wavelengths

        complex_array = np.array(complex_array)  # Refractive index array
        d_list = np.array(d_list)  # Thickness array
        d_list = d_list.astype(np.float64)  # Convert from string to float
        sun_incident = np.array(
            sun_incident[72][:, 0]
        )  # Ensure sun_incident is a numpy array
        optical_wavelengths = np.array(optical_wavelengths)

        N = np.array(complex_array, dtype=np.complex128)  # Convert N to numpy array
        T = np.array(d_list, dtype=np.float64)  # Convert T to numpy array
        N = np.swapaxes(
            N, 0, 1
        )  # Swap axes so that layers come first (N shape should be [8, 38])

        O = tmm.coh_tmm(
            "s", N, T, Theta, optical_wavelengths
        )  # Assuming vectmm.coh_tmm works with numpy arrays
        t = O["T"]  # Transmission powers
        r = O["R"]  # Reflection powers
        t_amp = np.sqrt(t)
        r_amp = np.sqrt(r)

        return t_amp, r_amp

    def t_grid(self, incident_grid, optical_wavelengths, n_list, d_list):

        t_grid_frames = []

        for i in range(len(incident_grid)):

            t_grid = np.empty(incident_grid[0].shape, dtype=object)

            for j in range(t_grid.shape[1]):
                # Refractive index array
                complex_array = np.array(n_list)

                # Thickness array
                d_list = np.array(d_list)

                # Convert from string to float
                d_list = d_list.astype(np.float64)

                # Ensure sun_incident is a numpy array
                sun_incident = np.array(incident_grid[i][:, j])
                optical_wavelengths = np.array(optical_wavelengths)

                # Assuming complex_array is your refractive index
                # array (N) and d_list is the thickness array (T)

                # Convert N to numpy array
                N = np.array(complex_array, dtype=np.complex128)
                # Convert T to numpy array
                T = np.array(d_list, dtype=np.float64)

                # Swap the axes of N to have shape [8, 38] instead of [38, 8]
                # Swap axes so that layers come first (N shape should be [8, 38])
                N = np.swapaxes(N, 0, 1)

                # Incident angles and wavelengths as numpy arrays
                # Incident angles (Theta)
                Theta = np.array(sun_incident, dtype=np.float64)

                # Wavelengths
                optical_wavelengths = np.array(optical_wavelengths, dtype=np.float64)

                # If you were previously using torch-based functions, replace
                # them with numpy equivalents or modify the function you're
                # using accordingly.

                # Assuming vectmm.coh_tmm works with numpy arrays
                O = tmm.coh_tmm("s", N, T, Theta, optical_wavelengths)

                # Transmission power
                t = O["T"]
                # Transmission amplitude
                t_amp = np.sqrt(t)

                # t_grid[:, j] = t_amp
                for k in range(t.shape[0]):
                    # t[k] is a 1D array of length 38
                    t_grid[k, j] = t_amp[k]

            # Append the t_grid for this frame to the list of frames
            t_grid_frames.append(t_grid)

        return t_grid_frames

    def solar_cells_irradiance_rays(
        self, solar_intensity_frames, t_grid_frames, solar_cell_exposure_maps
    ):
        # Initialize an empty list to store modified t_grid
        modified_spectra_grid_frames = []

        # Loop through each frame in t_grid
        for i in range(len(t_grid_frames)):
            t_grid_frame = t_grid_frames[i]
            # Initialize the modified t_grid for this frame
            modified_t_grid = np.empty_like(t_grid_frame, dtype=object)

            # Loop through each cell in the frame
            for j in range(t_grid_frame.shape[0]):
                for k in range(t_grid_frame.shape[1]):
                    # Get the transmission array for the current cell
                    transmittance = t_grid_frame[j, k]

                    # Get the solar spectrum and exposure map values
                    spectrum = solar_intensity_frames[i]
                    exposure_map = solar_cell_exposure_maps[j, k]

                    # Check if the exposure map has a 1 (i.e., apply the spectrum)
                    if exposure_map == 1:
                        # Multiply transmittance by the solar spectrum
                        modified_t_grid[j, k] = transmittance * spectrum
                    else:
                        # Return the solar spectrum directly if exposure map is not 1
                        modified_t_grid[j, k] = spectrum

            # Append the modified frame to the list
            modified_spectra_grid_frames.append(modified_t_grid)

        return modified_spectra_grid_frames

    def direct_irradiance_spectra_ground(
        self,
        ground_grid,
        normals_unit_ground,
        surface_grid,
        distances_grid,
        sun_vecs,
        transmitted_spectra_frames,
    ):

        ground_shape = (len(distances_grid), len(distances_grid[0]))
        irradiance_traced = []

        for i in range(len(transmitted_spectra_frames)):
            irradiance_grid = np.zeros(ground_shape, dtype=object)

            # Calculate intersections and closest points
            for j in range(ground_shape[0]):
                for k in range(ground_shape[1]):
                    # Get ground point
                    x_g = ground_grid[0][j][k]
                    y_g = ground_grid[1][j][k]
                    z_g = ground_grid[2][j][k]

                    # Normalize sun vector
                    sun_vec = np.array(sun_vecs[i])
                    sun_vec = sun_vec / np.linalg.norm(sun_vec)

                    # Compute t for intersection with the cylindrical surface
                    # a = sun_vec[0]**2 + sun_vec[2]**2
                    # b = 2 * (x_g * sun_vec[0] + z_g * sun_vec[2])
                    # c = x_g**2 + z_g**2 - self.radius**2
                    # Compute t for intersection with the cylindrical surface (elliptical shape)
                    a = (sun_vec[0] / self.radius1) ** 2 + (
                        sun_vec[2] / self.radius2
                    ) ** 2
                    b = 2 * (
                        x_g * sun_vec[0] / (self.radius1) ** 2
                        + z_g * sun_vec[2] / (self.radius2) ** 2
                    )
                    c = x_g**2 / (self.radius1) ** 2 + z_g**2 / (self.radius2) ** 2 - 1

                    discriminant = b**2 - 4 * a * c

                    if discriminant >= 0:
                        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

                        # Choose the correct t (positive and smallest)
                        t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)

                        # Calculate intersection point on the cylindrical surface
                        x_int = x_g + t * sun_vec[0]
                        y_int = y_g + t * sun_vec[1]
                        z_int = z_g + t * sun_vec[2]

                        # Check if the intersection is within the tunnel bounds
                        if 0 <= y_int <= self.length:
                            # Find the closest surface grid point
                            distances = np.sqrt(
                                (surface_grid[0] - x_int) ** 2
                                + (surface_grid[1] - y_int) ** 2
                                + (surface_grid[2] - z_int) ** 2
                            )
                            closest_j, closest_k = np.unravel_index(
                                np.argmin(distances), distances.shape
                            )
                            irradiance_array = transmitted_spectra_frames[i][closest_j][
                                closest_k
                            ]
                        else:
                            irradiance_array = np.zeros_like(
                                transmitted_spectra_frames[i][closest_j][closest_k]
                            )

                    else:
                        irradiance_array = np.zeros_like(
                            transmitted_spectra_frames[i][closest_j][closest_k]
                        )

                    # Apply transmissivity and set the irradiance in the grid
                    irradiance_grid[j][k] = irradiance_array * np.dot(
                        sun_vecs[i], normals_unit_ground[:, j, k]
                    )

            irradiance_traced.append(irradiance_grid)

        return irradiance_traced

    def diffuse_irradiance_spectra_ground(
        self,
        distances_grid,
        separation_unit_vector_grid,
        normals_unit_ground,
        normals_unit_surface,
        surface_areas,
        irradiance_frames,
    ):

        diffuse_irradiance_frames = []

        ground_shape = (len(distances_grid), len(distances_grid[0]))

        for j in range(len(irradiance_frames)):
            irradiance_ground = np.zeros(ground_shape, dtype=object)

            # Iterate over each point on the ground grid
            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):

                    total_irradiance = 0.0

                    # Iterate over each point on the surface grid
                    surface_shape = irradiance_frames[j].shape
                    for k in range(surface_shape[0]):
                        for l in range(surface_shape[1]):
                            # Sum the irradiance contribution from each surface point
                            distance = distances_grid[p][q][k][l]
                            ground_projection = np.dot(
                                normals_unit_ground[:, p, q],
                                separation_unit_vector_grid[p][q][k][l],
                            )
                            surface_projection = np.dot(
                                normals_unit_surface[:, k, l],
                                separation_unit_vector_grid[p][q][k][l],
                            )

                            irradiance = (
                                ground_projection
                                * surface_projection
                                * surface_areas[k][l]
                                * irradiance_frames[j][k][l]
                                / (2 * np.pi * (distance) ** 2)
                            )
                            total_irradiance += irradiance

                    # Store the result in the irradiance_ground array
                    irradiance_ground[p][q] = total_irradiance

            diffuse_irradiance_frames.append(irradiance_ground)

        return diffuse_irradiance_frames

    def global_irradiance_spectra_ground(
        self, direct_irradiance_spectra_ground, diffuse_irradiance_spectra_ground
    ):

        global_irradiance_frames = []
        global_photon_frames = []

        ground_shape = (
            len(direct_irradiance_spectra_ground[0]),
            len(direct_irradiance_spectra_ground[0][0]),
        )

        for i in range(len(direct_irradiance_spectra_ground)):
            irradiance_ground = np.zeros(ground_shape, dtype=object)

            for p in range(ground_shape[0]):
                for q in range(ground_shape[1]):

                    total_irradiance = (
                        direct_irradiance_spectra_ground[i][p][q]
                        + diffuse_irradiance_spectra_ground[i][p][q]
                    )

                    irradiance_ground[p][q] = total_irradiance

            global_irradiance_frames.append(irradiance_ground)

        return global_irradiance_frames

    # def power_to_photon_spectra(self, wavelengths_sample, power_spectra):

    #     photon_spectra = np.zeros_like(power_spectra) #demand photon spectra
    #     h = 6.63e-34
    #     c = 3e8
    #     N = 6.02e23
    #     hcN = ( 6.63 * 3 * 6.02 ) * 1e-3

    #     # Iterate over the indices of irradiance_frames_spectra and shaded_exposure_maps
    #     for i in range(len(power_spectra)):
    #         for j in range(len(power_spectra[i])):
    #                 for k in range(len(power_spectra[i][j])):
    #                     # If exposure map is 1, copy the entire array at irradiance_frames_spectra[i][j][k]
    #                     # photon_spectra[i][j][k] = power_spectra[i][j][k] * (wavelengths_sample/h*c) * N * 1e6
    #                     photon_spectra[i][j][k] = power_spectra[i][j][k] * (wavelengths_sample/( h * c )) * N * 1e6 # umol
    #                     # photon_spectra[i][j][k] = power_spectra[i][j][k] * wavelengths_sample / (h * c * N)

    #     return photon_spectra

    def power_to_photon_spectra(self, wavelengths_sample, power_spectra):
        """
        Power spectrum to a photon flux spectrum in micromoles (μmol).

        Args:
            wavelengths_sample (array): Wavelengths (in nanometers).
            power_spectra (array): Multidimensional array of spectral power (W/m²/nm) per time

        Returns:
            np.ndarray: Array of the same shape as power_spectra, representing photon flux in μmol/s/m²/nm.
        """
        # Physical constants
        h = 6.63e-34  # Planck's constant (J·s)
        c = 3e8  # Speed of light (m/s)
        N = 6.02e23  # Avogadro's number (1/mol)

        # Convert wavelengths from nanometers to meters
        wavelengths_m = wavelengths_sample * 1e-9  # nm to m

        # Initialize the photon spectra array
        photon_spectra = np.zeros_like(power_spectra)

        # Conversion from Watts (W) to micromoles of photons (μmol):
        # The formula used is:
        # photon_flux (μmol) = power (W) * (λ / (h * c)) * N * 1e6
        # Where:
        # - λ: wavelength in meters (m)
        # - h: Planck's constant (J·s)
        # - c: speed of light (m/s)
        # - N: Avogadro's number (mol⁻¹)
        # - 1e6: conversion factor from mol to μmol

        # Iterate over all indices in the power_spectra array
        for i in range(len(power_spectra)):
            for j in range(len(power_spectra[i])):
                for k in range(len(power_spectra[i][j])):
                    photon_spectra[i][j][k] = (
                        power_spectra[i][j][k] * (wavelengths_m[k] / (h * c * N)) * 1e6
                    )
                    # photon_spectra[i][j][k] = ( power_spectra[i][j][k] * (wavelengths_sample[k] / (h * c * N)) * 1e6 )

        return photon_spectra

    def par_spectra(self, wavelengths_sample, photon_spectra):

        # L# Read CSV file into a DataFrame
        file_path = os.path.join(
            "..", "data", "PAR-quantum-yield-spectrum-Inada-1976.csv"
        )
        df = pd.read_csv(file_path, skipinitialspace=True)

        # Extract relevant columns
        wavelengths_n_nm = df["λ,n (nm)"].values
        par = df["PAS"].values

        wavelengths_sample = np.array(wavelengths_sample)
        par = np.array(par)

        # Perform interpolation
        int_par_data = np.interp(wavelengths_sample, wavelengths_n_nm, par)

        par_ammended_spectra = np.zeros_like(photon_spectra)  # demand photon spectra

        # Iterate over the indices of irradiance_frames_spectra and shaded_exposure_maps
        for i in range(len(photon_spectra)):
            for j in range(len(photon_spectra[i])):
                for k in range(len(photon_spectra[i][j])):
                    # If exposure map is 1, copy the entire array at irradiance_frames_spectra[i][j][k]
                    par_ammended_spectra[i][j][k] = (
                        photon_spectra[i][j][k] * int_par_data
                    )

        return par_ammended_spectra

    def int_spectra(self, wavelengths, spectra_frames):

        int_frames = []

        for i in range(len(spectra_frames)):
            grid_frame = spectra_frames[i]
            # Initialize the modified t_grid for this frame
            int_grid = np.empty_like(grid_frame, dtype=object)

            for j in range(int_grid.shape[0]):
                for k in range(int_grid.shape[1]):

                    integral = trapezoid(spectra_frames[i][j][k], wavelengths, 0.01)

                    int_grid[j, k] = integral

            int_frames.append(int_grid)

        return int_frames

    def scaled_spectra_sensor_bf5(self, wavelengths, spectra_frames):
        scaled_frames = []

        # spectral_response_sensor_bf5: srbf5
        file_srbf5 = os.path.join("..", "data", "clean_bf5_spectral_response.csv")
        srbf5 = pd.read_csv(file_srbf5)
        srbf5_wavelength = srbf5["sx"].to_numpy()
        srbf5_response = srbf5["sy"].to_numpy()

        interpolated_srbf5 = np.interp(wavelengths, srbf5_wavelength, srbf5_response)

        for iter_frame, frame in enumerate(spectra_frames):
            for j in range(frame.shape[0]):
                for k in range(frame.shape[0]):
                    frame[j][k] = frame[j][k] * interpolated_srbf5

        return spectra_frames
