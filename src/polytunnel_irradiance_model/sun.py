import pandas as pd
import numpy as np
import pvlib 
from pvlib import spectrum, solarposition, irradiance, atmosphere
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import trapezoid


class Sun:
    def __init__(self, start_time='2024-07-30T00:00:00Z', end_time='2024-07-30T23:59:59Z', latitude = 51.1950, longitude = 0.2757, altitude=0, resolution_minutes=1):
        self.start_time = self.parse_datetime(start_time)
        self.end_time = self.parse_datetime(end_time)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.resolution = resolution_minutes
    
    @staticmethod
    def parse_datetime(datetime_str):
        return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ')

    def get_times(self):
        times = []
        current_time = self.start_time
        while current_time <= self.end_time:
            times.append(current_time)
            current_time += timedelta(minutes=self.resolution)
        return times

    def generate_sun_positions(self):
        times = self.get_times()
        
        # Create a DataFrame for the times
        times_df = pd.DatetimeIndex(times)

        # Get solar position data
        solar_position = pvlib.solarposition.get_solarposition(times_df, self.latitude, self.longitude, altitude=self.altitude)

        # Extract altitude and azimuth
        altitude_array = solar_position['apparent_elevation'].values
        azimuth_array = solar_position['azimuth'].values
        
        return altitude_array, azimuth_array
    
    def generate_sun_distance(self):
        times = self.get_times()
        
        # Create a DataFrame for the times
        times_df = pd.DatetimeIndex(times)

        # Get solar position data
        distance_array = pvlib.solarposition.nrel_earthsun_distance(times_df).values
        
        return distance_array

    def generate_sun_vecs(self, sun_positions):
       
        sun_vecs = []
        for altitude, azimuth in sun_positions:
            azimuth_rad = np.radians(azimuth)
            altitude_rad = np.radians(altitude)
            sun_dir_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
            sun_dir_y = np.cos(azimuth_rad) * np.cos(altitude_rad)
            sun_dir_z = np.sin(altitude_rad)
            sun_vec = np.array([sun_dir_x, sun_dir_y, sun_dir_z])
            sun_vecs.append(sun_vec)

        return sun_vecs
    
    def sunvec_tilts_grid(self, sun_vecs, normals_grid):
        
        angle_grid_frames = []
        scalar_grid_frames = []

        for i in range(len(sun_vecs)):
            sun_vec = sun_vecs[i]

            scalar_grid = np.zeros((len(normals_grid[0]), len(normals_grid[0][0])))
            angle_grid = np.zeros((len(normals_grid[0]), len(normals_grid[0][0])))

            # Iterate through each (i, j) index in the normals_grid
            for j in range(len(normals_grid[0][0])):
                for k in range(len(normals_grid[0][0])):
                    # Perform dot product between sun_vec and each normal vector at normals_grid[i][j]
                    vec = np.array([normals_grid[0][j][k], normals_grid[1][j][k], normals_grid[2][j][k]])
                    scalar = np.dot(sun_vec, vec)
                    scalar = np.clip(scalar, -1.0, 1.0)
                    scalar_grid[j][k] = scalar
                    angle_grid[j][k] = np.arccos(scalar)

            scalar_grid_frames.append(scalar_grid)
            angle_grid_frames.append(angle_grid)

        return scalar_grid_frames, angle_grid_frames
    
    def generate_sun_vec_grids(self, sun_vecs, surface_grid, distance_array):
        
        X, Y, Z = surface_grid
        X_array = []
        Y_array = []
        Z_array = []

        for i in range(len(distance_array)):
            sun_pos = sun_vecs[i] * distance_array[i] * 1.496e+8
            new_X = sun_pos[0] - X
            new_Y = sun_pos[1] - Y
            new_Z = sun_pos[2] - Z
            
            magnitude = np.sqrt(new_X**2 + new_Y**2 + new_Z**2)

            norm_X = new_X / magnitude
            norm_Y = new_Y / magnitude
            norm_Z = new_Z / magnitude

            X_array.append(norm_X) 
            Y_array.append(norm_Y) 
            Z_array.append(norm_Z)

        return X_array, Y_array, Z_array

    def get_spectra(self, integration_region_q1, integration_region_q2):
  # assumptions from the technical report:
        times = self.get_times()
        lat = self.latitude
        lon = self.longitude
        tilt = 0
        azimuth = 180
        pressure = 101300  # sea level, roughly
        water_vapor_content = 0.5  # cm
        tau500 = 0.1
        ozone = 0.31  # atm-cm
        albedo = 0.2
        
        solar_spectra_frames = []
        spectral_irradiance_frames = []
        for i in range(len(times)):
            solpos = solarposition.get_solarposition(times[i], lat, lon)
            aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)

            # The technical report uses the 'kasten1966' airmass model, but later
            # versions of SPECTRL2 use 'kastenyoung1989'.  Here we use 'kasten1966'
            # for consistency with the technical report.
            relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith,
                                                            model='kasten1966')
            
            spectra = spectrum.spectrl2(
                apparent_zenith=solpos.apparent_zenith,
                aoi=aoi,
                surface_tilt=tilt,
                ground_albedo=albedo,
                surface_pressure=pressure,
                relative_airmass=relative_airmass,
                precipitable_water=water_vapor_content,
                ozone=ozone,
                aerosol_turbidity_500nm=tau500
            )
            wavelengths = spectra['wavelength']
            intensities = spectra['poa_global'].flatten()
        
            optical_mask = (wavelengths >= integration_region_q1) & (wavelengths <= integration_region_q2)
            optical_wavelengths = wavelengths[optical_mask]
            optical_intensities = np.nan_to_num(intensities[optical_mask], nan=0.0)
 
            integral = trapezoid(optical_intensities, optical_wavelengths, 0.01)
            solar_spectra_frames.append(optical_intensities)
            spectral_irradiance_frames.append(integral)

        return optical_wavelengths, solar_spectra_frames, spectral_irradiance_frames
    