########################################################################################
# solar.py --- Polytunnel solar module for the Polytunnel-Irradiance Module.           #
#                                                                                      #
# Author(s): Taylor Pomfret, Emilio Nunez-Andrade, Benedict Winchester                 #
# Date created: Summer 2024/25                                                         #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `solar.py`

This module contains information used for solar position and irradiance calculations.

"""

import datetime

from dataclasses import dataclass
from math import cos, pi, radians, sin
from typing import Any, Generator, TypeVar

import pandas as pd
import numpy as np
import pvlib

from pvlib import spectrum, solarposition, irradiance, atmosphere, location
from scipy.integrate import trapezoid

from src.polytunnel_irradiance_model.__utils__ import Location
from src.polytunnel_irradiance_model.polytunnel import Vector


__all__ = ("SolarPosition",)

# APPARENT_ELEVATION:
#   Keyword for parsing information returned from pvlib about the apparent elevation of
# the sun.
APPARENT_ELEVATION: str = "apparent_elevation"

# APPARENT_ELEVATION:
#   Keyword for parsing information returned from pvlib about the azimuthal position of
# the sun.
AZIMUTH: str = "azimuth"


@dataclass
class SolarPosition:
    """
    Contains information about the position of the sun at a given time.

    .. attribute:: azimuthal_angle:
        The azimuthal position of the sun, in degrees.

    .. attribute:: elevation:
        The elevation of the sun above the horizon, in degrees.

    """

    azimuthal_angle: float
    elevation: float


# Type variable for Meshpoint and children.
_SPV = TypeVar(
    "_SPV",
    bound="SolarPositionVector",
)


@dataclass
class SolarPositionVector(Vector):
    """
    Contains information about the position of the sun at a given time.

    .. attribute:: azimuthal_angle:
        The azimuthal position of the sun, in degrees.

    .. attribute:: elevation:
        The elevation of the sun above the horizon, in degrees.

    """

    azimuthal_angle: float
    elevation: float

    @classmethod
    def from_solar_position(cls, solar_position: SolarPosition) -> _SPV:
        """
        Instantiate a vector, with additional information, from the solar position.

        :param: solar_position:
            The solar position, as a :class:`SolarPosition` instance.

        :returns:
            An instantiated :class:`SolarPositionVector` instance.

        """

        z_coord: float = cos(
            (theta_radians := pi / 2 - radians(solar_position.elevation))
        )
        x_coord: float = sin(theta_radians) * cos(
            phi_radians := radians(solar_position.azimuthal_angle)
        )
        y_coord: float = sin(theta_radians) * sin(phi_radians)

        return cls(
            x_coord,
            y_coord,
            z_coord,
            solar_position.azimuthal_angle,
            solar_position.elevation,
        )


def calculate_solar_position(
    location: Location,
    time: datetime.datetime | list[datetime.datetime] | pd.DatetimeIndex,
    *,
    altitude: float = 0,
) -> list[SolarPositionVector]:
    """
    Calculate the solar position based on the time.

    :param: location:
        The location.

    :param: time:
        The time.

    :param: altitude:
        The altitude above the ground.

    :returns:
        The position of the sun, as a :class:`SolarPosition` instance.

    """

    # Sanitise the time
    if not isinstance(time, pd.DatetimeIndex):
        if isinstance(time, list):
            time = pd.DatetimeIndex(time)
        else:
            time = pd.DatetimeIndex(
                [
                    time,
                ]
            )

    # Compute the solar positions
    solar_position = pvlib.solarposition.get_solarposition(
        time, location.latitude, location.longitude, altitude
    )

    # Return the results of the calculation
    return [
        SolarPositionVector.from_solar_position(
            SolarPosition(entry[AZIMUTH], entry[APPARENT_ELEVATION])
        )
        for _, entry in solar_position.iterrows()
    ]


def calculate_clearsky_data_new(
    location: Location,
    time_iterator: Generator[datetime.datetime, Any, None],
) -> pd.DataFrame:
    """
    Retrieves clear-sky irradiance data (GHI and DNI) for a given date range.

    The function utilises the in-built functionality of :class:`pvlib.location.Location`
    instances to compute the clearsky irradiance at a location for a series of times.

    :param: location:
        The :class:`Location` instance being modelled.

    :param: time_iterator:
        A :class:`Generator` instance which is capable of iterating through the times
        and yielding times when needed.

    :returns:
        A :class:`pd.DataFrame` instance containing information about the irradiance
        information at each time step.

    """

    return location.location.get_clearsky(pd.to_datetime(list(time_iterator)))


class Sun:
    def __init__(
        self,
        start_time="2024-07-30T00:00:00Z",
        end_time="2024-07-30T23:59:59Z",
        latitude=51.1950,
        longitude=0.2757,
        altitude=0,
        resolution_minutes=1,
    ):
        self.start_time = self.parse_datetime(start_time)
        self.end_time = self.parse_datetime(end_time)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.resolution = resolution_minutes

    @staticmethod
    def parse_datetime(datetime_str):
        return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")

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
        solar_position = pvlib.solarposition.get_solarposition(
            times_df, self.latitude, self.longitude, altitude=self.altitude
        )

        # Extract altitude and azimuth
        altitude_array = solar_position["apparent_elevation"].values
        azimuth_array = solar_position["azimuth"].values

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
            # for j in range(len(normals_grid[0][0])): # Changes to adapt to differnt tile points
            for j in range(len(normals_grid[0])):
                for k in range(len(normals_grid[0][0])):
                    # Perform dot product between sun_vec and each normal vector at normals_grid[i][j]
                    vec = np.array(
                        [
                            normals_grid[0][j][k],
                            normals_grid[1][j][k],
                            normals_grid[2][j][k],
                        ]
                    )
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
            sun_pos = sun_vecs[i] * distance_array[i] * 1.496e8
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
            relative_airmass = atmosphere.get_relative_airmass(
                solpos.apparent_zenith, model="kasten1966"
            )

            spectra = spectrum.spectrl2(
                apparent_zenith=solpos.apparent_zenith,
                aoi=aoi,
                surface_tilt=tilt,
                ground_albedo=albedo,
                surface_pressure=pressure,
                relative_airmass=relative_airmass,
                precipitable_water=water_vapor_content,
                ozone=ozone,
                aerosol_turbidity_500nm=tau500,
            )
            wavelengths = spectra["wavelength"]
            intensities = spectra["poa_global"].flatten()

            optical_mask = (wavelengths >= integration_region_q1) & (
                wavelengths <= integration_region_q2
            )
            optical_wavelengths = wavelengths[optical_mask]
            optical_intensities = np.nan_to_num(intensities[optical_mask], nan=0.0)

            integral = trapezoid(optical_intensities, optical_wavelengths, 0.01)
            solar_spectra_frames.append(optical_intensities)
            spectral_irradiance_frames.append(integral)

        return optical_wavelengths, solar_spectra_frames, spectral_irradiance_frames

    def get_clearsky_data_new(
        self,
        start_time_str,
        end_time_str,
        latitude=51.1950,
        longitude=0.2757,
        res_minutes=1,
    ):
        """
        Obtiene los datos de irradiancia de cielo despejado (GHI y DNI) para un rango de fechas dado.

        Args:
            start_time_str (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
            end_time_str (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
            latitude (float): Latitud de la localización. Por defecto es 51.1950.
            longitude (float): Longitud de la localización. Por defecto es 0.2757.
            res_minutes (int): Resolución temporal en minutos para resamplear los datos. Por defecto es 1.

        Returns:
            pd.DataFrame: DataFrame con las columnas ['ghi', 'dni', 'ghi_clear', 'dni_clear'].
        """
        # Convertir las fechas de inicio y fin a formato datetime
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)

        # Crear un objeto Location con la ubicación
        tz = "Europe/London"
        site = location.Location(latitude, longitude, tz=tz)

        # Crear un DatetimeIndex para el rango de fechas con la resolución deseada
        times = pd.date_range(
            start=start_time, end=end_time, freq=f"{res_minutes}T", tz="UTC"
        )

        # Calcular la irradiancia de cielo despejado (GHI y DNI)
        clear_sky = site.get_clearsky(times)

        # Resamplear a la resolución temporal deseada si es necesario
        cs_resampled = clear_sky.resample(f"{res_minutes}T").mean()

        return cs_resampled

    def calculate_diffuse_irradiance_meshgrid(self, ghi, dhi, normals_unit_surface):
        """
        Calcula la irradiancia difusa sobre cada superficie del meshgrid.

        Args:
            ghi (pd.Series): Irradiancia global horizontal (GHI).
            dhi (pd.Series): Irradiancia difusa horizontal (DHI).
            normals_unit_surface (np.ndarray): Normales unitarias de las superficies del túnel (Nx3 array).

        Returns:
            diffuse_irradiance_grid (np.ndarray): Irradiancia difusa sobre cada superficie (Nx1 array).
        """
        # Asegurar que las dimensiones sean compatibles
        normals_unit_surface = np.array(
            normals_unit_surface
        )  # Normales unitarias (Nx3)
        num_surfaces = normals_unit_surface.shape[0]

        # Inicializar irradiancia difusa sobre cada superficie
        diffuse_irradiance_grid = np.zeros(
            (len(dhi), num_surfaces)
        )  # DHI por superficie para todos los tiempos

        for t_idx, dhi_t in enumerate(dhi):  # Iterar sobre cada instante de tiempo
            for i in range(num_surfaces):  # Iterar sobre cada superficie
                # Suposición isotrópica: factor de corrección basado en el ángulo de inclinación
                normal = normals_unit_surface[i]  # Normal de la superficie
                cos_theta_surface = max(
                    0, normal[2]
                )  # Componente Z de la normal (para inclinación horizontal)

                # Distribuir irradiancia difusa
                diffuse_irradiance_grid[t_idx, i] = dhi_t * cos_theta_surface

        return diffuse_irradiance_grid
