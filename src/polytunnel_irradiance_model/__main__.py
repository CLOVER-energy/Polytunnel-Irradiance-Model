########################################################################################
# __main__.py --- Main entry point for running the irradiance model from the CLI.      #
#                                                                                      #
# Author(s): Taylor Pomfret, Emilio Nunez-Andrade, Benedict Winchester                 #
# Date created: Summer 2024                                                            #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `__main__.py`

The model functions to compute, utilising spectral ray-tracing tools, the irradiance
distribution within a curved structure, _e.g._, a polytunnel.

"""

import argparse
import colour.plotting
import csv
import datetime
import enum
import functools
import os
import re
import subprocess
import sys
import time

from collections import defaultdict
from collections.abc import Sequence
from colour import SpectralDistribution, XYZ_to_sRGB, sd_to_XYZ
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, cos, floor, pi, sqrt
from typing import Any, Callable, Generator, Iterator, Match, Pattern

import json
import matplotlib.colors as m_colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
import yaml

from matplotlib import rc, rcParams
from numpy import inf
from scipy import constants
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.polytunnel_irradiance_model.__utils__ import *
from src.polytunnel_irradiance_model.functions import *
from src.polytunnel_irradiance_model.plotting import *
from src.polytunnel_irradiance_model.polytunnel import (
    calculate_adjacent_polytunnel_shading,
    calculate_adjacent_polytunnel_solid_angle_as_function_of_theta,
    calculate_and_update_intercept_planes,
    calculate_solid_angles,
    EndType,
    MeshPoint,
    NotInterceptError,
    Polytunnel,
    Plane,
    solid_angle_weighted_tmm,
)
from src.polytunnel_irradiance_model.solar import (
    calculate_solar_position,
    calculate_clearsky_data_new,
    SolarPositionVector,
)
from src.polytunnel_irradiance_model.irradiance import (
    ground_direct_irradiance,
    open_end_direct_irradiance,
)
from src.polytunnel_irradiance_model.tracing import Tracing
import src.polytunnel_irradiance_model.visualisation as viz


import warnings

import numpy as np

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r".*vectorized_tmm_dispersive_multistack.*"
)
warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ("compute_surface_grid", "main")

__version__ = "1.0.0a1"

# Plotting context
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 7


# AUTO_GENERATED:
#   Name for the auto generated--files directory.
AUTO_GENERATED: str = "auto_generated"

# BOLOMETER_ERROR:
#   The error in the bolometer as a fraction.
BOLOMETER_ERROR: float = 0.1

# CHLOROPHYL_A_EXTINCTION_FILENAME:
#   Name of the file containing chlorophyl-a extinction data
CHLOROPHYL_A_EXTINCTION_FILENAME: str = "chrlorophyl_a_extinction.csv"

# CLOUDY_DAY_SPECTRA_DATA_FILE:
#   Name of the file used for cloudy-day weather data.
CLOUDY_DAY_SPECTRA_DATA_FILE: str = "brecl_spectra_on_typical_days.csv"

# DIFFUSE_PAR_ERROR:
#   The fractional error in the diffuse PAR.
DIFFUSE_PAR_ERROR: float = 0.12

# DONE:
#   Snippet to print when code is succesffully executed.
DONE: str = "[  DONE  ]"

# FAILED:
#   Snippet to print when code fails to execute.
FAILED: str = "[ FAILED ]"

# HADLOW_WEATHER_FILENAME:
#   The Hadlow weather filename.
HADLOW_WEATHER_FILENAME: str = "hadlow_combined.csv"

# INCOMING_SHORTWAVE:
#   Keyword for parsing Hadlow's incoming shortwave information.
INCOMING_SHORTWAVE: str = "SWIN_LEVEL3"

# INCOMING_SHORTWAVE_DIFFUSE:
#   Keyword for diffuse component of incoming light.
INCOMING_SHORTWAVE_DIFFUSE: str = "SWIN_LEVEL3_DIFFUSE"

# INCOMING_SHORTWAVE_DIRECT:
#   Keyword for direct component of incoming light.
INCOMING_SHORTWAVE_DIRECT: str = "SWIN_LEVEL3_DIRECT"

# INDEX:
#   Integer for storing index of plots.
INDEX: int = 5

# MODULES:
#   Keyword for parsing PV-module information.
MODULES: str = "modules"

# MM:
#   Conversion factor from mm to inches.
MM: float = 1 / 25.4

# PAR_WAVELENGTH_RANGE:
#   The range of photosynthetically-active radiation.
PAR_WAVELENGTH_RANGE: list[float] = list(range(400, 701))

# POLYTUNNEL_HEADER_STRING:
#   Header string for the polytunnel-PV code.
POLYTUNNEL_HEADER_STRING: str = """
                                   #         #        #
                                   ###       ##     ###
                            ##      ##################      #
                             ####  #########################
                                ##########################
                          #######################################
                              ##############################
                              ##############################
                                               #############
                                                    ########
                              ################           #
                          #######           #####
 ###   ###              #####     ########     #####
 ##### ###             ###      #####  #######    ####
 ### #####            ###          ########         #####
 ###   ###           ###                   #################
                     ##               #######             ####
        ##            ###          #####         #######    #####
        #####          ###       ####        ######      ###   ####
         #######        ####    ###                #########      ####
          #####           #######                ###                 ####
          ## ####          #####                       #####################
              ####           ###                   ######              #######
                ####           ###             ######                        #####
                  ###           ####         #####                             #####
                   ####           ###       ###                                   ###
                     ###           ####    ###                                     ####
                      ####           ###  ##                                        ###
                        ###           ######                                         ###
                         ####           ###                                          ###
                           ####
                            ####
                              ###
                               ####
                                 ####
                                  ###

{version_line}
                              Polytunnel-Irradiance-Model
    An open-source modelling framework for the simulation and optimisation of curved
                      photovoltaic panels in agricultural contexts

                             For more information, contact
                  Benedict Winchester (benedict.winchester@gmail.com)
"""

# PYRANOMETER_RESPONSE_FILENAME:
#   The name of the file containing the pyranometer response data.
PYRANOMETER_RESPONSE_FILENAME: str = "pyranometer_nr01_ra01_response.csv"

# TOTAL_PAR_ERROR:
#   The fractional error in the total (global) PAR.
TOTAL_PAR_ERROR: float = 0.12

# VERSION_REGEX:
#   Regex used to extract the main version number.
VERSION_REGEX: Pattern[str] = re.compile(r"(?P<number>\d\.\d\.\d)([\.](?P<post>.*))?")

# WAVELENGTH:
#   Keyword for parsing wavelength information.
WAVELENGTH: str = "wavelength"


class Dashes(Iterator):
    """
    Contains the dash information for plotting.

    .. attribute:: dashes
        The dashes to include.

    """

    def __init__(self) -> None:
        """
        Instantiate the dashes.

        """

        self.dash_length: int = 0
        self.dot_length: int = 0
        self.dotted: bool = False
        self.space: int = 1
        super().__init__()

    def __next__(self) -> Generator[tuple[int, int], None, None]:
        """
        Iterate through and yield the dash information.

        :yields: The dash information as a `tuple`.

        """

        # Move on the dash length
        self.dash_length += 1

        # Move on the space length if needed
        if self.dash_length - self.space > 3:
            self.dash_length = 1
            self.space += 1

        # If needed, add dots.
        if self.space > 5:
            self.dotted = True
            self.dash_length = 0
            self.space = 1

        if self.dotted:
            self.dot_length += 1
            return (self.dash_length, self.space, self.dot_length, self.space)

        return (self.dash_length, self.space)


class DummyTMM(Sequence):
    """
    Represents a TMM when no TMM is required.

    .. attribute:: length:
        The length of the TMM.

    .. attribute:: transmittance:
        The transmittance through the TMM for all wavelengths.

    """

    def __init__(self, length: int, transmittance: float = 0.0):
        """
        Instantiate a :class:`DummyTMM` instance.

        :param: length:
            The length of the TMM.

        :param: transmittance:
            The transmittance through the TMM to use for all wavelengths.

        """

        self.length = length
        self._series: pd.Series | None = None
        self.transmittance: float = transmittance
        super().__init__()

    def __getitem__(self, i):
        """Returns a series mocking the item."""
        if self._series is None:
            self._series = pd.Series([self.transmittance] * len(self))
        return self._series

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        """
        Return a nice-looking string representing the TMM.

        :returns:
            A nice-looking string representing the TMM.

        """

        return f"DummyTmm(transmittance={self.transmittance})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def columns(self) -> pd.Index:
        """
        Return mocked columns.

        :returns:
            Mocked columns.

        """

        return pd.Index([0])


class ValidationColumns(enum.Enum):
    """
    Contains the names of the column headers in the validationd data.

    - DIFFUSE_PAR:
        The diffuse PAR information

    - DIRECT_PAR:
        The direct PAR illumination.

    - LABEL:
        The name of the label.

    - SECTION:
        The name of the section being used.

    - TOTAL_PAR:
        The total PAR illumination.

    """

    DIFFUSE_ERROR: str = "diffuse std"
    DIFFUSE_PAR: str = "diffuse illum umol  m2 -1 s -1"
    DIRECT_ERROR: str = "direct std"
    DIRECT_PAR: str = "direct illum umol  m2 -1 s -1"
    LABEL: str = "Label"
    SECTION: str = "Section"
    TOTAL_ERROR: str = "total std"
    TOTAL_PAR: str = "total illum umol m2 -1 s -1"


class SpectrumType(enum.Enum):
    """
    Denotes the type of spectrum being modelled.

    - CLEARSKY_DIFFUSE:
        Denotes a clearsky, diffuse spectrum.

    - CLEARSKY_DIRECT:
        Denotes a clearsky, direct spectrum.

    - CLEARSKY_GLOBAL:
        Denotes a clearsky, global spectrum.

    - CLOUDY_DAY:
        Denotes a cloudy-day spectrum.

    """

    CLEARSKY_DIFFUSE: str = "diffuse"
    CLEARSKY_DIRECT: str = "direct"
    CLEARSKY_GLOBAL: str = "global"
    CLOUDY_DAY: str = "cloudy_day"


class SpectrumUnit(enum.Enum):
    """
    Contains the possible spectrum units for the cloudy-day spectra.

    - W_PER_M2_NM:
        Watts per meter squared per nanometer.

    - W_PER_M2_UM:
        Watts per meter squared per micrometer.

    """

    W_PER_M2_NM = "watts_per_m2_per_nm"
    W_PER_M2_UM = "watts_per_m2_per_um"


def code_print(string_to_print: str, end: str = "") -> None:
    """
    Print a line with dots.

    :param: string_to_print:
        The string to print.

    """

    print(string_to_print + "." * (64 - len(string_to_print)), end="")


def _yield_time(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    timestep: datetime.timedelta,
) -> Generator[datetime.datetime, Any, None]:
    """
    Yield times within the simulation.

    :yields:
        Times within the range specified.

    """

    this_time = start_time
    while end_time > this_time:
        yield this_time
        this_time += timestep


def parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parse the CLI arguments.

    :param: args:
        The unparsed CLI arguments.

    :return: The parsed arguments as a :class:`argparse.Namespace`

    """

    parser = argparse.ArgumentParser(description="Ray-tracer for PV Polytunnel")

    # Simulation arguments
    simulation_arguments = parser.add_argument_group(
        "simulation arguments", description="Arguments used for running a simulation."
    )
    simulation_arguments.add_argument(
        "--start-time",
        "-st",
        type=str,
        default="2024-07-30T00:00:00Z",
        help="The start-time string, in a YYYY-MM-DDTHH:MM:SSZ format, with literal T"
        " and Z characters.",
    )
    simulation_arguments.add_argument(
        "--end-time",
        "-et",
        type=str,
        default="2024-07-30T23:59:59Z",
        help="The end-time string, in a YYYY-MM-DDTHH:MM:SSZ format, with literal T"
        " and Z characters.",
    )

    simulation_arguments.add_argument(
        "--latitude",
        "-lat",
        type=float,
        default=51.251841,
        help="The latitude of the location for which weather data should be used.",
    )
    simulation_arguments.add_argument(
        "--longitude",
        "-lon",
        type=float,
        default=0.347040,
        help="The longitude of the location for which weather data should be used.",
    )
    simulation_arguments.add_argument(
        "--altitude",
        "-alt",
        type=float,
        default=0,
        help="The altitude of the location for which weather data should be used.",
    )
    simulation_arguments.add_argument(
        "--modelling-temporal-resolution",
        "-mtr",
        type=float,
        default=30,
        help="The temporal resolution, in minutes, to use when simulating throughout "
        "the day.",
    )
    simulation_arguments.add_argument(
        "--skip-animations",
        "-sa",
        action="store_true",
        default=False,
        help="Flag to skip plotting animations.",
    )
    simulation_arguments.add_argument(
        "--skip-plots",
        "-sp",
        action="store_true",
        default=False,
        help="Flag to skip plotting and exit.",
    )
    simulation_arguments.add_argument(
        "--validation-filename",
        "-vf",
        type=str,
        default=None,
        help="The name of the validation file to use.",
    )
    simulation_arguments.add_argument(
        "--validation-index",
        "-vi",
        type=int,
        default=None,
        help="The validation element to use.",
    )

    # Polytunnel arguments
    polytunnel_arguments = parser.add_argument_group(
        "polytunnel arguments",
        description="Arguments used to specify the technical details of the polytunnel.",
    )
    polytunnel_arguments.add_argument(
        "--diffusivity",
        "-d",
        type=float,
        default=None,
        help="The diffusivity of the polytunnel material to use.",
    )
    polytunnel_arguments.add_argument(
        "--polytunnel-input-file",
        "-pif",
        type=str,
        default="polytunnels.yaml",
        help="The name of the polytunnels input file to use.",
    )
    polytunnel_arguments.add_argument(
        "--polytunnel",
        "-pt",
        type=str,
        help="The name of the polytunnel to use.",
    )

    # Solar cell arguments.
    solar_cell_arguments = parser.add_argument_group(
        "solar-cell arguments",
        description="Arguments for specifying the configuration of the PV cells/modules.",
    )
    solar_cell_arguments.add_argument(
        "--solar-cells-file",
        "--materials-file",
        type=str,
        default="solar_cells.yaml",
        help="The path to the solar-cells materials inputs file.",
    )
    # parser.add_argument(
    #     "--initial_cell_spacing",
    #     type=float,
    #     default=0.0,
    #     help="initial_cell_spacing (default: 1.0 (1 meter)                  )",
    # )

    parser.add_argument(
        "--meshgrid-resolution",
        "-mres",
        type=int,
        default=10,
        help="The resolution of the mesh grid in terms of the number of points along "
        "each dimension of the polytunnel to use; default of 10.",
    )
    parser.add_argument(
        "--weather-file",
        "-wf",
        type=str,
        default=HADLOW_WEATHER_FILENAME,
        help="The name of the weather-data file to use.",
    )
    parser.add_argument(
        "--weather-file-error",
        "-wfer",
        type=float,
        default=0.38,
        help="The fractional error of the data contained in the weather-data file.",
    )
    parser.add_argument(
        "--weather-as-diffusivity-only",
        "-wado",
        action="store_true",
        default=False,
        help="Use the alternative weather-data file for diffusivity data only.",
    )
    parser.add_argument(
        "--hadlow-weather-filename",
        "-hwf",
        type=str,
        default=HADLOW_WEATHER_FILENAME,
        help=argparse.SUPPRESS,
    )

    tmm_and_spectral_arguments = parser.add_argument_group("solar-spectra arguments")
    tmm_and_spectral_arguments.add_argument(
        "--cloudy-day-spectra-data-file",
        "-cdsdf",
        type=str,
        default=CLOUDY_DAY_SPECTRA_DATA_FILE,
        help="Name of the data file used for cloudy-day solar spectra.",
    )
    tmm_and_spectral_arguments.add_argument(
        "--cloudy-day-spectra-units",
        "-cdsu",
        type=str,
        default=SpectrumUnit.W_PER_M2_UM,
        choices=[entry.value for entry in SpectrumUnit],
        help="Unit for the spectral units.",
    )
    tmm_and_spectral_arguments.add_argument(
        "--wavelength-step-nm",
        "-wsnm",
        type=float,
        default=1,
        help="Resolution to use for TMM and wavelength calculations in nm.",
    )
    tmm_and_spectral_arguments.add_argument(
        "--regenerate-tmm",
        "-rtm",
        default=False,
        action="store_true",
        help="Regenerate the polytunnel solar TMMs.",
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        default=False,
        help="Regenerate surface-irradiance plots.",
    )
    parser.add_argument(
        "--regenerate-mesh",
        action="store_true",
        default=False,
        help="Regenerate the polytunnel-to-ground mesh.",
    )

    parser.add_argument(
        "--debug", "-dbug", action="store_true", default=False, help=argparse.SUPPRESS
    )

    return parser.parse_args(args)


def round_nearest(x: float, a: float):
    """
    Function to round the nearest number to a decimal or other float.

    :param: x:
        The number to round.

    :param: a:
        The number to round to the nearest multiple of.

    :returns:
        The result of the rounding process.

    """

    return round(x / a) * a


@contextmanager
def time_execution(
    this_code_block_name: str,
) -> Generator[Callable[[], float], Any, None]:
    """
    Times a period of code execution.

    :yields:
        The elapsed time taken for the code to execute.

    """

    # code_print(this_code_block_name, end="\r")
    start_time: float = time.perf_counter()
    end_time: float | None = None

    try:
        yield lambda: (
            end_time - start_time
            if end_time is not None
            else time.perf_counter() - start_time
        )
    except Exception:
        code_print(this_code_block_name, end="")
        print(f"{'.' * 13} {FAILED}")
        raise
    else:
        code_print(this_code_block_name, end="")
        execution_time: str = str(round(time.perf_counter() - start_time, 3))
        print(f"{'.' * (10 - len(execution_time))} {execution_time} s {DONE}")
    finally:
        end_time = time.perf_counter()


def main(args: list[Any]) -> None:
    """
    Main function for operating the Polytunnel-Irradiance-Model.

    :param: args:
        The unparsed command-line arguments.

    """

    # Snippet taken with permission from CLOVER-energy/CLOVER
    # >>>
    version_match: Match[str] | None = VERSION_REGEX.match(__version__)
    version_number: str = (
        version_match.group("number") if version_match is not None else __version__
    )
    version_string = f"Version {version_number}"
    print(
        POLYTUNNEL_HEADER_STRING.format(
            version_line=(
                " " * (44 - ceil(len(version_string) / 2))
                + version_string
                + " " * (44 - floor(len(version_string) / 2))
            )
        )
    )
    # <<< end of reproduced snippted

    # Parse the command-line arguments.
    parsed_args = parse_args(args)

    # Open the material information.
    with open(parsed_args.solar_cells_file, "r", encoding="UTF-8") as solar_cells_file:
        material_information = yaml.safe_load(solar_cells_file)

    pv_module_inputs = {entry[NAME]: entry for entry in material_information[MODULES]}

    # Open the polytunnels information.
    with open(
        parsed_args.polytunnel_input_file, "r", encoding="UTF-8"
    ) as polytunnel_file:
        polytunnel_information = yaml.safe_load(polytunnel_file)

    # Assert that a polytunnel was specified.
    try:
        polytunnel_data = {
            polytunnel_data[NAME]: polytunnel_data
            for polytunnel_data in polytunnel_information["polytunnels"]
        }[parsed_args.polytunnel]
    except KeyError:
        raise KeyError(
            "Missing polytunnel information. Check all information in the file is "
            "correct and that the name specified on the command line matches a "
            "polytunnel defined in the inputs file."
        )

    # Compute the simulation date and time.
    simulation_start_datetime = datetime.datetime.strptime(
        parsed_args.start_time, "%Y-%m-%dT%H:%M:%SZ"
    )
    simulation_end_datetime = datetime.datetime.strptime(
        parsed_args.end_time, "%Y-%m-%dT%H:%M:%SZ"
    )

    if not os.path.isdir(
        output_figures_dir := os.path.join(
            "output_files", "figures", simulation_start_datetime.strftime("%Y_%m_%d")
        )
    ):
        os.makedirs(output_figures_dir)

    # Carry out the Polytunnel geometry instantiation calculation.
    with time_execution("Polytunnel geometry calculation"):
        polytunnel: Polytunnel = Polytunnel.from_data(
            polytunnel_data, parsed_args.meshgrid_resolution, pv_module_inputs
        )

        polytunnel_surface_pv_uncovered_fraction_map = pd.DataFrame(
            [1 - meshpoint.covered_fraction for meshpoint in polytunnel.surface_mesh]
        )

    # Open the TMM and, if necessary, compute.
    wavelength_step_nm: float | int = parsed_args.wavelength_step_nm
    _tmm_angular_resolution: float | int = 1
    if float(wavelength_step_nm) == int(wavelength_step_nm):
        wavelength_step_nm = int(wavelength_step_nm)
    if polytunnel.pv_module is not None and polytunnel.pv_module.stack is not None:
        # If there is no stack file, then compute in Julia.
        if (
            not os.path.isfile(
                stack_filename := f"{polytunnel.pv_module.stack_name}.csv"
            )
            or parsed_args.regenerate_tmm
        ):
            code_print("Running JULIA computation for stack")
            run = subprocess.run(
                command := (
                    f"julia tmm_ppv_script.jl -s {polytunnel.pv_module.stack_name} -t "
                    f"0:{_tmm_angular_resolution}:90 "
                    f"-f {polytunnel.pv_module.stack_name} -w {wavelength_step_nm}"
                ).split(" ")
            )
            if run.returncode != 0:
                raise Exception(
                    "TMM code failed: check STDOUT. "
                    f"Command sent to Julia: {command}"
                )
            print(DONE)

        with open(stack_filename, "r", encoding="UTF-8") as stack_tmm_file:
            stack_tmm: pd.DataFrame | DummyTMM | None = pd.read_csv(stack_tmm_file)

        stack_tmm = stack_tmm.set_index(WAVELENGTH)
        stack_tmm.columns = pd.Index([float(entry) for entry in stack_tmm.columns])

    else:
        stack_tmm = None

    # Load the solar spectra from the data files.
    try:
        with open(
            parsed_args.cloudy_day_spectra_data_file, "r", encoding="UTF-8"
        ) as cloudy_day_spectra_data_file:
            cloudy_day_spectra = pd.read_csv(cloudy_day_spectra_data_file)

    except FileNotFoundError:
        raise FileNotFoundError(
            "Cloudy day--spectra data file ({filename}) was not found.".format(
                filename=parsed_args.cloudy_day_spectra_data_file
            )
        ) from None

    # Pop metadata rows until the data are only spectral and not metadata.
    cloudy_day_spectra = cloudy_day_spectra.transpose()

    def _recursive_row_removal(frame: pd.DataFrame, index: int = 0) -> pd.DataFrame:
        """
        Recursively remove rows from the dataframe.

        :param: frame:
            The :class:`pd.DataFrame` to recursively remove rows from.

        :returns:
            The frame with only values and no metadata remaining.

        """

        try:
            return frame.astype(float).transpose()
        except ValueError:
            frame.pop(index)
            return _recursive_row_removal(frame, index + 1)

    cloudy_day_spectra = (
        _recursive_row_removal(cloudy_day_spectra)
        .reset_index(drop=True)
        .set_index(WAVELENGTH)
    )

    if (
        spectra_units := SpectrumUnit(parsed_args.cloudy_day_spectra_units)
    ) == SpectrumUnit.W_PER_M2_UM:
        cloudy_day_spectra /= 1000
        spectra_units = SpectrumUnit.W_PER_M2_NM

    # Load the reference solar spcetrum from PVlib for a sunny day.
    reference_day_spectra = pvlib.spectrum.get_reference_spectra()

    # Recompute the spectra at wavelength steps matching those used in the TMM
    # calculation.
    import numpy as np

    if isinstance(stack_tmm, pd.DataFrame):
        wavelength_range = np.arange(
            stack_tmm.index[0],
            stack_tmm.index[-1] + wavelength_step_nm,
            wavelength_step_nm,
        )

    # If no TMMs are needed, because no PV modules are included, then use the reference
    # limits on spectra.
    else:
        wavelength_range = np.arange(
            reference_day_spectra.index[0],
            reference_day_spectra.index[-1] + wavelength_step_nm,
            wavelength_step_nm,
        )
        stack_tmm = DummyTMM(len(wavelength_range))

    global_spectrum = [
        interp1d(
            reference_day_spectra.index,
            reference_day_spectra[SpectrumType.CLEARSKY_GLOBAL.value],
            fill_value=(0, 0),
            bounds_error=False,
        )(entry)
        for entry in wavelength_range
    ]
    direct_spectrum = [
        interp1d(
            reference_day_spectra.index,
            reference_day_spectra[SpectrumType.CLEARSKY_DIRECT.value],
            fill_value=(0, 0),
            bounds_error=False,
        )(entry)
        for entry in wavelength_range
    ]
    cloudy_spectrum = [
        interp1d(
            cloudy_day_spectra.index,
            cloudy_day_spectra[SpectrumType.CLOUDY_DAY.value],
            fill_value=(0, 0),
            bounds_error=False,
        )(entry)
        for entry in wavelength_range
    ]
    interpolated_spectra = pd.DataFrame(
        {
            SpectrumType.CLEARSKY_DIFFUSE.value: pd.Series(global_spectrum)
            - pd.Series(direct_spectrum),
            SpectrumType.CLEARSKY_DIRECT.value: direct_spectrum,
            SpectrumType.CLEARSKY_GLOBAL.value: global_spectrum,
            SpectrumType.CLOUDY_DAY.value: cloudy_spectrum,
        }
    )
    interpolated_spectra.index = pd.Index(wavelength_range, name=WAVELENGTH)

    # Normalise the interpolated spectra to 1 over the wavelength range.
    interpolated_spectra /= interpolated_spectra.sum(axis=0)

    # Rescale the spectra by the pyranometer response integrated over this range vs over
    # the whole datarange.
    if os.path.isfile(PYRANOMETER_RESPONSE_FILENAME):
        with time_execution("Rescaling spectra with pyranometer spectral response"):
            with open(
                PYRANOMETER_RESPONSE_FILENAME, "r", encoding="UTF-8"
            ) as pyranometer_response_file:
                pyranometer_response_data: pd.DataFrame = pd.read_csv(
                    pyranometer_response_file
                )

            pyranometer_wavelength_range = range(0, 10000)
            pyranometer_response_data = pyranometer_response_data.set_index(WAVELENGTH)
            pyranometer_response_function = interp1d(
                pyranometer_response_data.index.values,
                pyranometer_response_data.values,
                axis=0,
                fill_value=(0, 0),
                bounds_error=False,
            )
            pyranometer_response = [
                pyranometer_response_function(wavelength)[0]
                for wavelength in pyranometer_wavelength_range
            ]
            pyranometer_response /= max(pyranometer_response)

            # Compute and multiply the pyranometer response by the global irradiance at each wavelength.
            _global_spectrum = [
                interp1d(
                    reference_day_spectra.index,
                    reference_day_spectra[SpectrumType.CLEARSKY_GLOBAL.value],
                    fill_value=(0, 0),
                    bounds_error=False,
                )(entry)
                for entry in pyranometer_wavelength_range
            ]
            _global_spectrum /= sum(_global_spectrum)
            pyranometer_global_response_list = (
                (_global_spectrum * np.array(pyranometer_response))
                .astype(float)
                .tolist()
            )
            adjusted_global_spectrum = _global_spectrum / sum(
                pyranometer_global_response_list
            )

            _direct_spectrum = [
                interp1d(
                    reference_day_spectra.index,
                    reference_day_spectra[SpectrumType.CLEARSKY_DIRECT.value],
                    fill_value=(0, 0),
                    bounds_error=False,
                )(entry)
                for entry in pyranometer_wavelength_range
            ]
            _direct_spectrum /= sum(_direct_spectrum)
            pyranometer_direct_response_list = (
                (_direct_spectrum * np.array(pyranometer_response))
                .astype(float)
                .tolist()
            )
            adjusted_direct_spectrum = _direct_spectrum / sum(
                pyranometer_direct_response_list
            )

            _diffuse_spectrum = [
                interp1d(
                    interpolated_spectra.index,
                    interpolated_spectra[SpectrumType.CLEARSKY_DIFFUSE.value],
                    fill_value=(0, 0),
                    bounds_error=False,
                )(entry)
                for entry in pyranometer_wavelength_range
            ]
            _diffuse_spectrum /= sum(_diffuse_spectrum)
            pyranometer_diffuse_response_list = (
                _diffuse_spectrum * np.array(pyranometer_response)
            ).tolist()
            adjusted_diffuse_spectrum = _diffuse_spectrum / sum(
                pyranometer_diffuse_response_list
            )

            _cloudy_spectrum = [
                interp1d(
                    cloudy_day_spectra.index,
                    cloudy_day_spectra[SpectrumType.CLOUDY_DAY.value],
                    fill_value=(0, 0),
                    bounds_error=False,
                )(entry)
                for entry in pyranometer_wavelength_range
            ]
            _cloudy_spectrum /= sum(_cloudy_spectrum)
            pyranometer_cloudy_day_response_list = (
                _cloudy_spectrum * np.array(pyranometer_response)
            ).tolist()
            adjusted_cloudy_spectrum = _cloudy_spectrum / sum(
                pyranometer_cloudy_day_response_list
            )

            #######################
            # Plotting code No. 0 #
            #######################

            # Construct a frame which can be used.
            pyranometer_adjusted_interpolated_spectra = pd.DataFrame(
                {
                    WAVELENGTH: pyranometer_wavelength_range,
                    SpectrumType.CLEARSKY_DIFFUSE.value: adjusted_diffuse_spectrum,
                    SpectrumType.CLEARSKY_DIRECT.value: adjusted_direct_spectrum,
                    SpectrumType.CLEARSKY_GLOBAL.value: adjusted_global_spectrum,
                    SpectrumType.CLOUDY_DAY.value: adjusted_cloudy_spectrum,
                }
            )
            pyranometer_adjusted_interpolated_spectra = (
                pyranometer_adjusted_interpolated_spectra.set_index(WAVELENGTH)
            )
            pyranometer_adjusted_interpolated_spectra = (
                pyranometer_adjusted_interpolated_spectra.loc[wavelength_range]
            )

    else:
        pyranometer_adjusted_interpolated_spectra = interpolated_spectra.copy()

    # Compute the position of the sun at each time within the simulation.
    location = Location(
        parsed_args.altitude, parsed_args.latitude, parsed_args.longitude
    )

    with time_execution("Solar position calculation"):
        solar_positions = calculate_solar_position(
            location,
            list(
                _yield_time(
                    simulation_start_datetime,
                    simulation_end_datetime,
                    datetime.timedelta(
                        minutes=parsed_args.modelling_temporal_resolution
                    ),
                )
            ),
            altitude=location.altitude,
        )
        polytunnel_surface_pv_uncovered_fraction_mask = pd.concat(
            [polytunnel_surface_pv_uncovered_fraction_map.transpose()]
            * len(solar_positions),
            axis=0,
        )

    # Compute the clearsky irradiance at the location, using the solar spectrum for a
    # clearsky day.
    with time_execution("Clearsky irradiance calculation"):
        clearsky_irradiance = calculate_clearsky_data_new(
            location,
            _yield_time(
                simulation_start_datetime,
                simulation_end_datetime,
                datetime.timedelta(minutes=parsed_args.modelling_temporal_resolution),
            ),
        )

    # Compute weather-realted parameters
    with time_execution("Weather calculation"):
        # Read the Hadlow weather data if available.
        try:
            with open(
                os.path.join(parsed_args.hadlow_weather_filename), "r", encoding="UTF-8"
            ) as hadlow_weather_file:
                hadlow_weather_data: pd.DataFrame = (
                    pd.read_csv(hadlow_weather_file)
                    .drop([0, 1])
                    .set_index("parameter-id")
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find weather file: {parsed_args.hadlow_weather_filename}"
            ) from None

        hadlow_weather_data.index = pd.DatetimeIndex(hadlow_weather_data.index)
        hadlow_weather_data = hadlow_weather_data.tz_localize(None)

        # Slice using the start and end weather values.
        hadlow_dni_slice: pd.DataFrame = hadlow_weather_data[
            parsed_args.start_time.replace("T", " ")
            .replace("Z", "") : parsed_args.end_time.replace("T", " ")
            .replace("Z", "")
        ][INCOMING_SHORTWAVE].astype(float)

        # Resample based on the modelling resolution provided.
        hadlow_dni_slice = hadlow_dni_slice[
            [
                entry.minute % int(parsed_args.modelling_temporal_resolution) == 0
                for entry in hadlow_dni_slice.index
            ]
        ]

        # If the CLI has been used to specify an additional weather file, use this in
        # addition to locally-obtained data.
        if parsed_args.weather_file != parsed_args.hadlow_weather_filename:
            try:
                with open(
                    os.path.join(parsed_args.weather_file), "r", encoding="UTF-8"
                ) as weather_file:
                    alternative_weather_data: pd.DataFrame | None = (
                        pd.read_csv(weather_file).drop([0, 1]).set_index("utc_time")
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find weather file: {parsed_args.weather_file}"
                ) from None

            alternative_weather_data.index = pd.DatetimeIndex(
                alternative_weather_data.index
            )
            alternative_weather_slice = alternative_weather_data[
                parsed_args.start_time.replace("T", " ")
                .replace("Z", "") : parsed_args.end_time.replace("T", " ")
                .replace("Z", "")
            ].astype(float)

            # Create a label used for distinguishing that alternative weather data has
            # been used.
            alt_weather: str = "alt_weather_"

        else:
            alternative_weather_data = None
            alt_weather: str = ""

        # If the weather data file is to be used only for diffusivity data, then
        # use the Hadlow data for the real weather values. Otherwise, use these data
        if (parsed_args.weather_file != parsed_args.hadlow_weather_filename) and (
            not parsed_args.weather_as_diffusivity_only
        ):
            dhi_to_weather_adjustment_factor: pd.DataFrame = pd.concat(
                [
                    (
                        alternative_weather_slice.reset_index(drop=True)[
                            INCOMING_SHORTWAVE_DIRECT
                        ]
                        / (
                            _merged_frame := (
                                pd.merge(
                                    alternative_weather_slice,
                                    clearsky_irradiance,
                                    left_index=True,
                                    right_index=True,
                                ).reset_index(drop=True)
                            )
                        )["dhi"]
                    )
                    .clip(0, None)
                    .replace([inf, -inf], 0)
                ]
                * len(polytunnel.surface_mesh),
                axis=1,
            )
            dhi_to_weather_adjustment_factor.index = alternative_weather_slice.index

            dni_to_weather_adjustment_factor: pd.DataFrame = pd.concat(
                [
                    (
                        alternative_weather_slice.reset_index(drop=True)[
                            INCOMING_SHORTWAVE_DIFFUSE
                        ]
                        / _merged_frame["dni"]
                    )
                    .clip(0, None)
                    .replace([inf, -inf], 0)
                ]
                * len(polytunnel.surface_mesh),
                axis=1,
            )
            dni_to_weather_adjustment_factor.index = alternative_weather_slice.index

        else:
            dhi_to_weather_adjustment_factor: pd.DataFrame = pd.concat(
                [
                    (hadlow_dni_slice / clearsky_irradiance["dhi"])
                    .clip(0, None)
                    .replace([inf, -inf], 0)
                ]
                * len(polytunnel.surface_mesh),
                axis=1,
            )
            dni_to_weather_adjustment_factor: pd.DataFrame = pd.concat(
                [
                    (hadlow_dni_slice / clearsky_irradiance["dni"])
                    .clip(0, None)
                    .replace([inf, -inf], 0)
                ]
                * len(polytunnel.surface_mesh),
                axis=1,
            )

    # Make the auto-generated directory.
    os.makedirs(AUTO_GENERATED, exist_ok=True)
    os.makedirs(os.path.join(AUTO_GENERATED, polytunnel.name), exist_ok=True)

    if (
        not os.path.isfile(
            surface_shaded_map_filename := os.path.join(
                AUTO_GENERATED,
                polytunnel.name,
                f"{polytunnel.name}_surface_shaded_map_"
                f"{parsed_args.start_time.replace(":","_")}_"
                f"{parsed_args.end_time.replace(":","_")}.csv",
            )
        )
        or parsed_args.regenerate
    ):
        with time_execution("Surface shading calculation"):
            # Determine whether any of the modules are shaded by neighbouring polytunnels,
            # either in terms of the direct or diffuse contributions of light that they receive.
            #
            surface_shaded_map = pd.DataFrame(
                {
                    meshpoint_index: [
                        bool(
                            not calculate_adjacent_polytunnel_shading(
                                meshpoint, polytunnel, solar_position
                            )
                            and ((meshpoint.normal_vector * solar_position) > 0)
                            and (solar_position.elevation > 0)
                        )
                        for solar_position in solar_positions
                    ]
                    for meshpoint_index, meshpoint in tqdm(
                        enumerate(polytunnel.surface_mesh),
                        desc="Surface shading calculation",
                        leave=False,
                        total=len(polytunnel.surface_mesh),
                    )
                }
            )

        with open(
            surface_shaded_map_filename, "w", encoding="UTF-8"
        ) as surface_shaded_file:
            surface_shaded_map.to_csv(surface_shaded_file)

    else:
        with time_execution("Opening surface-shading data"):
            with open(
                surface_shaded_map_filename, "r", encoding="UTF-8"
            ) as surface_shaded_file:
                surface_shaded_map = pd.read_csv(surface_shaded_file, index_col=0)

            surface_shaded_map.columns = pd.Index(
                [int(entry) for entry in surface_shaded_map.columns]
            )

    # Compute the solar position dot-product across the surface of the polytunnel.
    dot_product_map = pd.DataFrame(
        {
            meshpoint_index: [
                meshpoint.normal_vector * solar_position
                for solar_position in solar_positions
            ]
            for meshpoint_index, meshpoint in tqdm(
                enumerate(polytunnel.surface_mesh),
                desc="Surface dot-product calculation",
                leave=False,
                total=len(polytunnel.surface_mesh),
            )
        }
    )

    # Carry out a calculation if the outputs have not already been saved.
    if (
        not os.path.isfile(
            diffuse_surface_filename := os.path.join(
                AUTO_GENERATED,
                polytunnel.name,
                f"{polytunnel.name}_diffuse_surface_irradiance_"
                f"{parsed_args.start_time.replace(":","_")}_"
                f"{parsed_args.end_time.replace(":","_")}.csv",
            )
        )
        or parsed_args.regenerate
    ):

        # Determine the intercept lines with neighbouring polytunnels.
        # calculate_and_update_intercept_planes(polytunnel)
        with time_execution("Direct surface calculation"):
            # Construct a map of the surface irradiance on the polytunnel (direct) as a
            # function of time.
            direct_surface_irradiance = (
                (surface_shaded_map * dot_product_map)
                .mul(clearsky_irradiance["dni"].values, axis=0)
                .reset_index(drop=True)
            )

            #######################
            # Plotting code No. 1 #
            #######################

        # Calculate the amount of diffuse light reaching the ground.
        with time_execution("Diffuse surface calculation"):
            calculate_solid_angles(polytunnel.surface_mesh, polytunnel)
            diffuse_surface_irradiance = pd.DataFrame(
                {
                    meshpoint_index: meshpoint.solid_angle
                    * clearsky_irradiance["dhi"]
                    / (2 * pi)
                    for meshpoint_index, meshpoint in tqdm(
                        enumerate(polytunnel.surface_mesh),
                        desc="Diffuse surface calculation",
                        leave=False,
                        total=len(polytunnel.surface_mesh),
                    )
                }
            )

            #######################
            # Plotting code No. 1 #
            #######################

        with open(
            os.path.join(
                AUTO_GENERATED,
                polytunnel.name,
                f"{polytunnel.name}_diffuse_surface_irradiance_"
                f"{parsed_args.start_time.replace(":","_")}_"
                f"{parsed_args.end_time.replace(":","_")}.csv",
            ),
            "w",
            encoding="UTF-8",
        ) as output_file:
            diffuse_surface_irradiance.to_csv(output_file)

        with open(
            os.path.join(
                AUTO_GENERATED,
                polytunnel.name,
                f"{polytunnel.name}_direct_surface_irradiance_"
                f"{parsed_args.start_time.replace(":","_")}_"
                f"{parsed_args.end_time.replace(":","_")}.csv",
            ),
            "w",
            encoding="UTF-8",
        ) as output_file:
            direct_surface_irradiance.to_csv(output_file)

    else:
        with time_execution("Opening diffuse-surface data"):
            with open(
                diffuse_surface_filename,
                "r",
                encoding="UTF-8",
            ) as diffuse_surface_file:
                diffuse_surface_irradiance: pd.DataFrame = pd.read_csv(
                    diffuse_surface_file, index_col=0
                )

            diffuse_surface_irradiance.columns = pd.Index(
                [int(entry) for entry in diffuse_surface_irradiance.columns]
            )

        with time_execution("Opening direct-surface data"):
            with open(
                os.path.join(
                    AUTO_GENERATED,
                    polytunnel.name,
                    f"{polytunnel.name}_direct_surface_irradiance_"
                    f"{parsed_args.start_time.replace(":","_")}_"
                    f"{parsed_args.end_time.replace(":","_")}.csv",
                ),
                "r",
                encoding="UTF-8",
            ) as direct_surface_file:
                direct_surface_irradiance: pd.DataFrame = pd.read_csv(
                    direct_surface_file, index_col=0
                )

            direct_surface_irradiance.columns = pd.Index(
                [int(entry) for entry in direct_surface_irradiance.columns]
            )

    diffuse_surface_irradiance.index = dni_to_weather_adjustment_factor.index
    direct_surface_irradiance.index = dni_to_weather_adjustment_factor.index

    if (
        not os.path.isfile(
            mesh_mesh_filename := os.path.join(
                AUTO_GENERATED,
                polytunnel.name,
                f"{polytunnel.name}_{polytunnel.meshgrid_resolution}_by_"
                f"{polytunnel.length_wise_meshgrid_resolution}_mesh_mesh_distance.json",
            )
        )
        or parsed_args.regenerate_mesh
    ):
        with time_execution("Mesh-mesh distance calculation"):
            # FIXME: Check this!
            # Consider each point on the surface as imparting diffuse light on the ground.
            ground_to_surface_projection_map: defaultdict[int, dict[int, float]] = {
                ground_index: {
                    # Angle between vector from ground to surface, dotted with the normal to
                    # the ground;
                    surface_index: abs(
                        (_vector := (ground_meshpoint - surface_meshpoint))
                        * ground_meshpoint.normal_vector
                    )
                    # multiplied by the angle between the ground-to-surface veccto and the
                    # normal of the surface;
                    * abs(_vector * surface_meshpoint.normal_vector)
                    # multiplied by the area of the surface element to go from Watts to
                    # Watts per meter squared;
                    * surface_meshpoint.area
                    # all normalised by the 1/distance^2 to scale back to Watts/meter^2.
                    / (
                        abs(surface_meshpoint.normal_vector)
                        * abs(ground_meshpoint.normal_vector)
                        * abs(_vector) ** 4
                    )
                    for surface_index, surface_meshpoint in tqdm(
                        enumerate(polytunnel.surface_mesh),
                        desc=f"Point {ground_index} calculation",
                        leave=False,
                        total=len(polytunnel.surface_mesh),
                    )
                }
                for ground_index, ground_meshpoint in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Mesh-mesh distance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            }

            with open(mesh_mesh_filename, "w", encoding="UTF-8") as mesh_mesh_file:
                json.dump(ground_to_surface_projection_map, mesh_mesh_file)

    else:
        with time_execution("Opening mesh-mesh distance calculation"):
            with open(mesh_mesh_filename, "r", encoding="UTF-8") as mesh_mesh_file:
                ground_to_surface_projection_map = {
                    int(key): value for key, value in json.load(mesh_mesh_file).items()
                }

    ground_to_surface_projection_frame: pd.DataFrame = pd.DataFrame(
        ground_to_surface_projection_map
    )

    # Compute the angle of the sun for each element on the surface in degrees.
    solar_angles: pd.DataFrame = round_nearest(
        np.degrees(np.acos(dot_product_map)), _tmm_angular_resolution
    )
    surface_post_tmm_direct_spectra: np.ndarray = np.array(
        [
            [
                stack_tmm[min(angle, max(stack_tmm.columns))]
                * pyranometer_adjusted_interpolated_spectra.direct.values
                for angle in row
            ]
            for _, row in tqdm(
                solar_angles.iterrows(),
                desc="Post-TMM direct spectra calculation",
                total=len(solar_angles),
            )
        ]
    )

    with time_execution("Computing surface diffuse-light TMM"):
        obscured_solid_angle: list[defaultdict[float, float]] = (
            calculate_adjacent_polytunnel_solid_angle_as_function_of_theta(
                polytunnel.surface_mesh,
                polytunnel,
                functools.partial(round_nearest, a=_tmm_angular_resolution),
            )
        )
        diffuse_surface_tmm = (
            pd.concat(
                solid_angle_weighted_tmm(
                    meshpoints=polytunnel.surface_mesh,
                    obscured_solid_angle=obscured_solid_angle,
                    rounding_function=functools.partial(
                        round_nearest, a=_tmm_angular_resolution
                    ),
                    tmm=stack_tmm,
                    tmm_angular_resolution=_tmm_angular_resolution,
                ),
                axis=1,
            )
            .transpose()
            .to_numpy()[None, :, :]
        )

    with time_execution("Readjusting with Hadlow data"):
        # Without spectra:
        clearsky_total_diffuse_surface_irradiance = (
            diffuse_surface_irradiance.reset_index(drop=True)
            + (
                polytunnel_diffusivity := (
                    parsed_args.diffusivity
                    if parsed_args.diffusivity is not None
                    else polytunnel.diffusivity
                )
            )
            * direct_surface_irradiance.reset_index(drop=True)
        )

        diffuse_day_total_diffuse_surface_irradiance = (
            diffuse_surface_irradiance * dhi_to_weather_adjustment_factor
        )
        direct_day_total_diffuse_surface_irradiance = (
            clearsky_total_diffuse_surface_irradiance
            * dni_to_weather_adjustment_factor.reset_index(drop=True)
        )

        # With spectra:
        # Compute the uncovered regions' diffuse-day and direct-day contributions to the
        # surface diffuse irradiance.
        diffuse_day_total_diffuse_surface_irradiance_sans_pv = (
            polytunnel_surface_pv_uncovered_fraction_mask.reset_index(drop=True)
            * diffuse_surface_irradiance.reset_index(drop=True)  # [W/m^2]
            * dhi_to_weather_adjustment_factor.reset_index(drop=True)  # [W/m^2 / W/m^2]
        ).to_numpy()[:, :, None] * np.array(
            pyranometer_adjusted_interpolated_spectra.cloudy_day
        )[
            None, None, :
        ]  # [Dimensionless intensity]

        direct_day_total_diffuse_surface_irradiance_sans_pv = (
            polytunnel_surface_pv_uncovered_fraction_mask.reset_index(drop=True)
            * (
                diffuse_surface_irradiance.reset_index(drop=True)
                + polytunnel_diffusivity  # [Dimensionless]
                * direct_surface_irradiance.reset_index(drop=True)  # [W/m^2]
            )
            * dni_to_weather_adjustment_factor.reset_index(drop=True)  # [W/m^2 / W/m^2]
        ).to_numpy()[:, :, None] * np.array(
            pyranometer_adjusted_interpolated_spectra.diffuse
        )[
            None, None, :
        ]  # [Dimensionless intensity]

        # Compute the covered regions' contributions using a solid-angle integral over
        # the TMMs.
        direct_day_total_diffuse_surface_irradiance_with_pv = (
            # The contribution from direct light which passes through at its incident
            # angle and is then diffused.
            polytunnel_diffusivity
            * (1 - polytunnel_surface_pv_uncovered_fraction_mask.reset_index(drop=True))
            * (
                direct_surface_irradiance.reset_index(drop=True)  # [W/m^2]
                * dni_to_weather_adjustment_factor.reset_index(
                    drop=True
                )  # [Dimensionless]
            )
        ).to_numpy()[:, :, None] * surface_post_tmm_direct_spectra * np.array(
            pyranometer_adjusted_interpolated_spectra.direct  # [Dimensionless intensity]
        )[
            None, None, :
        ] + (
            # The contribution of diffuse light which passes through as diffuse light at
            # various angles.
            (
                (
                    1
                    - polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
                        drop=True
                    )
                )
                * (diffuse_surface_irradiance.reset_index(drop=True))  # [W/m^2]
                * dhi_to_weather_adjustment_factor.reset_index(
                    drop=True
                )  # [W/m^2 / W/m^2]
            ).to_numpy()[:, :, None]
            * np.array(pyranometer_adjusted_interpolated_spectra.diffuse)[
                None, None, :
            ]  # [Dimensionless intensity]
            * diffuse_surface_tmm  # [Dimless transmittance]
        )

        diffuse_day_total_diffuse_surface_irradiance_with_pv = (
            (
                (
                    1
                    - polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
                        drop=True
                    )
                )
                * diffuse_surface_irradiance.reset_index(drop=True)
                * dhi_to_weather_adjustment_factor.reset_index(drop=True)
            ).to_numpy()[:, :, None]
            * np.array(pyranometer_adjusted_interpolated_spectra.cloudy_day)[
                None, None, :
            ]
            * diffuse_surface_tmm
        )

        # Combine these for a direct day,
        direct_day_total_diffuse_surface_irradiance = (
            direct_day_total_diffuse_surface_irradiance_sans_pv
            + direct_day_total_diffuse_surface_irradiance_with_pv
        )
        # and a diffuse day,
        diffuse_day_total_diffuse_surface_irradiance = (
            diffuse_day_total_diffuse_surface_irradiance_sans_pv
            + diffuse_day_total_diffuse_surface_irradiance_with_pv
        )
        # and rescale for a clearsky day.
        clearsky_total_diffuse_surface_irradiance = (
            direct_day_total_diffuse_surface_irradiance
            / dhi_to_weather_adjustment_factor.to_numpy()[:, :, np.newaxis]
        )

        ########################
        # Plotting code No. 1d #
        ########################

    try:
        clearsky_total_diffuse_surface_irradiance.index = (
            dni_to_weather_adjustment_factor.index
        )
    except AttributeError:
        pass

    try:
        diffuse_day_total_diffuse_surface_irradiance.index = (
            dni_to_weather_adjustment_factor.index
        )
    except AttributeError:
        pass

    try:
        direct_day_total_diffuse_surface_irradiance.index = (
            dni_to_weather_adjustment_factor.index
        )
    except AttributeError:
        pass

    # Calculate the amount of polytunnel surface sunlight which will reach the ground,
    # both as diffuse and direct components.

    # Compute the amount of direct light reaching the ground.
    with time_execution("Clearsky direct on-the-ground calculation"):
        # Code without spectra
        clearsky_ground_direct_irradiance_map: pd.DataFrame = pd.DataFrame(
            [
                [
                    entry[0]
                    for entry in ground_direct_irradiance(
                        polytunnel.ground_mesh,
                        polytunnel,
                        (
                            surface_shaded_map.loc[time_index]
                            * clearsky_irradiance["dni"].iloc[time_index]
                        ).reset_index(drop=True)
                        * polytunnel_surface_pv_uncovered_fraction_mask.iloc[
                            time_index
                        ],
                        solar_position,
                        diffusivity=parsed_args.diffusivity,
                    )
                ]
                # for time_index, solar_position in tqdm(
                #     list(enumerate(solar_positions))[11:],
                #     desc="Clearsky direct ground irradiance calculation",
                #     leave=False,
                #     total=len(solar_positions[11:]),
                # )
                for time_index, solar_position in tqdm(
                    enumerate(solar_positions),
                    desc="Clearsky direct ground irradiance calculation",
                    leave=False,
                    total=len(solar_positions),
                )
            ]
        )

        # Code with spectra:
        # Compute direct light which doesn't pass through a solar module
        clearsky_ground_direct_irradiance_map_sans_pv_module: pd.DataFrame = (
            pd.DataFrame(
                [
                    [
                        entry[0]
                        for entry in ground_direct_irradiance(
                            polytunnel.ground_mesh,
                            polytunnel,
                            (
                                surface_shaded_map.loc[time_index]
                                * clearsky_irradiance["dni"].iloc[time_index]
                            ).reset_index(drop=True)
                            * polytunnel_surface_pv_uncovered_fraction_mask.iloc[
                                time_index
                            ],
                            solar_position,
                            # diffusivity=0,
                            diffusivity=parsed_args.diffusivity,
                        )
                    ]
                    for time_index, solar_position in tqdm(
                        enumerate(solar_positions),
                        desc="Non-PV direct ground irradiance calculation",
                        leave=False,
                        total=len(solar_positions),
                    )
                ]
            )
        )
        clearsky_ground_direct_irradiance_map_sans_pv_module: np.ndarray = (
            (
                clearsky_ground_direct_irradiance_map_sans_pv_module.to_numpy()[
                    :, :, None
                ]
                * np.array(pyranometer_adjusted_interpolated_spectra.direct)[
                    None, None, :
                ]
            )
            .astype(float)
            .clip(0, None)
        )

        ##########################################################
        # FIXME: Check that these spectra use the right indices. #
        ##########################################################

        # Compute the light which passes through the PV modules
        clearsky_ground_direct_irradiance_map_with_pv_module: pd.DataFrame = (
            pd.DataFrame(
                [
                    [
                        entry[0]
                        for entry in ground_direct_irradiance(
                            polytunnel.ground_mesh,
                            polytunnel,
                            (
                                surface_shaded_map.loc[time_index]
                                * clearsky_irradiance["dni"].iloc[time_index]
                            ).reset_index(drop=True)
                            * (
                                1
                                - polytunnel_surface_pv_uncovered_fraction_mask.iloc[
                                    time_index
                                ]
                            ),
                            solar_position,
                            # diffusivity=0,
                            diffusivity=parsed_args.diffusivity,
                        )
                    ]
                    for time_index, solar_position in tqdm(
                        enumerate(solar_positions),
                        desc="Through-PV direct ground irradiance calculation",
                        leave=False,
                        total=len(solar_positions),
                    )
                ]
            )
        ).clip(0, None)

        surface_index_of_illuminating_meshpoint: pd.DataFrame = (
            pd.DataFrame(
                [
                    [
                        entry[1]
                        for entry in ground_direct_irradiance(
                            polytunnel.ground_mesh,
                            polytunnel,
                            (
                                surface_shaded_map.loc[time_index]
                                * clearsky_irradiance["dni"].iloc[time_index]
                            ).reset_index(drop=True)
                            * (
                                1
                                - polytunnel_surface_pv_uncovered_fraction_mask.iloc[
                                    time_index
                                ]
                            ),
                            solar_position,
                            diffusivity=parsed_args.diffusivity,
                        )
                    ]
                    for time_index, solar_position in tqdm(
                        enumerate(solar_positions),
                        desc="Surface-index of PV calculation",
                        leave=False,
                        total=len(solar_positions),
                    )
                ]
            )
        ).clip(0, None)

        #######################
        # Plotting code No. 3 #
        #######################

        # Extract the surface spectra based on the meshpoint shining onto the ground.
        null_spectrum = 0 * np.array(pyranometer_adjusted_interpolated_spectra.direct)

        def _get_spectrum(surface_index: int | None, time_index: int) -> np.ndarray:
            """
            Get the spectrum from the surface based on the time and surface index.

            :param: surface_index:
                The index of the meshpoint on the surface casuing the illumination.

            :param: time_index:
                The index of the time of day.

            :return:
                The spectrum from the point.
            """
            if surface_index is None or np.isnan(surface_index):
                return null_spectrum
            return surface_post_tmm_direct_spectra[time_index, int(surface_index), :]

        ground_direct_irradiance_spectra = np.array(
            [
                [
                    _get_spectrum(surface_index, time_index)
                    for surface_index in surface_indices_row
                ]
                for time_index, surface_indices_row in surface_index_of_illuminating_meshpoint.reset_index(
                    drop=True
                ).iterrows()
            ]
        )

        # Compute the ground irradiance as based on the matching surface-mesh point.
        clearsky_ground_direct_irradiance_map_with_pv_module_and_spectra: np.ndarray = (
            clearsky_ground_direct_irradiance_map_with_pv_module.to_numpy()[:, :, None]
            * ground_direct_irradiance_spectra
        ).astype(float)

        clearsky_ground_direct_irradiance_map = (
            clearsky_ground_direct_irradiance_map_sans_pv_module
            + clearsky_ground_direct_irradiance_map_with_pv_module_and_spectra
        ).clip(0, None)

        # Plotting of spectra on the ground.
        #######################
        # Plotting code No. 4 #
        #######################

        # Plotting code of irradiance heatmaps on the ground.
        ########################
        # Plotting code No. 1b #
        ########################

        # If the ends are open, add the irradiance from the ends.
        if polytunnel.ends == EndType.OPEN:
            with time_execution("End--direct irradiance calculation"):
                end_intercept_projection: pd.DataFrame = pd.DataFrame(
                    [
                        open_end_direct_irradiance(
                            polytunnel.ground_mesh, polytunnel, solar_position
                        )
                        for solar_position in tqdm(
                            solar_positions,
                            desc="Light from ends",
                            leave=False,
                            total=len(solar_positions),
                        )
                    ]
                )

                # Code without spectra:
                end_direct_irradiance_map: pd.DataFrame | None = (
                    end_intercept_projection.transpose()
                    .mul(clearsky_irradiance["dni"].values)
                    .transpose()
                )

                # Code with spectra:
                end_direct_irradiance_map: np.ndarray = (
                    end_direct_irradiance_map.to_numpy()[:, :, None]
                    * np.array(pyranometer_adjusted_interpolated_spectra.direct)[
                        None, None, :
                    ]
                ).astype(float)

                with open(
                    os.path.join(
                        AUTO_GENERATED,
                        polytunnel.name,
                        f"{polytunnel.name}_end_irradiance_"
                        f"{parsed_args.start_time.replace(":","_")}_"
                        f"{parsed_args.end_time.replace(":","_")}.csv",
                    ),
                    "w",
                    encoding="UTF-8",
                ) as end_irradiance_file:
                    csv.writer(end_irradiance_file, delimiter=",").writerows(
                        end_direct_irradiance_map.tolist()
                    )
                    # end_direct_irradiance_map.to_csv(end_irradiance_file)

                #######################
                # Plotting code No. 1b #
                #######################

        clearsky_ground_direct_irradiance_map = (
            clearsky_ground_direct_irradiance_map
            * pd.DataFrame(
                [cos(position.theta_spherical) for position in solar_positions]
            )
            .reset_index(drop=True)
            .to_numpy()[:, :, None]
            + end_direct_irradiance_map.clip(0, None)
        )

        direct_day_ground_direct_irradiance = (
            clearsky_ground_direct_irradiance_map
            * dni_to_weather_adjustment_factor.reset_index(drop=True).to_numpy()[
                :, :, None
            ]
        )

        ########################
        # Plotting code No. 1b #
        ########################

        ########################
        # Plotting code No. 1c #
        ########################

    #######################
    # Plotting code No. 2 #
    #######################

    #######################
    # Plotting code No. 5 #
    #######################

    #######################
    # Plotting code No. 6 #
    #######################

    #######################
    # Plotting code No. 7 #
    #######################

    # Compute the amount of diffuse light reaching the ground.
    with time_execution("Diffuse on-the-ground calculation"):
        # Compute the diffuse irradiance on the ground.
        # Code without spectra.
        # clearsky_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
        #     {
        #         ground_index: (
        #             clearsky_total_diffuse_surface_irradiance.reset_index(drop=True)
        #             * polytunnel.transmissivity
        #             * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
        #                 drop=True
        #             )
        #             * ground_to_surface_projection_frame.iloc[ground_index]
        #         ).sum(axis=1)
        #         for ground_index, _ in tqdm(
        #             enumerate(polytunnel.ground_mesh),
        #             desc="Ground diffuse-irradiance calculation",
        #             leave=False,
        #             total=len(polytunnel.ground_mesh),
        #         )
        #     }
        # )
        # diffuse_day_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
        #     {
        #         ground_index: (
        #             diffuse_day_total_diffuse_surface_irradiance
        #             * polytunnel.transmissivity
        #             # * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
        #             #     drop=True
        #             # )
        #             * ground_to_surface_projection_frame.iloc[ground_index]
        #         ).sum(axis=1)
        #         for ground_index, _ in tqdm(
        #             enumerate(polytunnel.ground_mesh),
        #             desc="Diffuse day ground-irradiance calculation",
        #             leave=False,
        #             total=len(polytunnel.ground_mesh),
        #         )
        #     }
        # )

        # direct_day_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
        #     {
        #         ground_index: (
        #             direct_day_total_diffuse_surface_irradiance.reset_index(drop=True)
        #             * polytunnel.transmissivity
        #             * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
        #                 drop=True
        #             )
        #             * ground_to_surface_projection_frame.iloc[ground_index]
        #         ).sum(axis=1)
        #         for ground_index, _ in tqdm(
        #             enumerate(polytunnel.ground_mesh),
        #             desc="Direct day ground-irradiance calculation",
        #             leave=False,
        #             total=len(polytunnel.ground_mesh),
        #         )
        #     }
        # )sly

        # Code with spectra.
        # Compute the diffuse irradiance on the ground for clearsky conditions whereby
        # the spectra are determined based on clearsky conditions and no adjustment for
        # DHI or DNI is done.
        clearsky_ground_diffuse_irradiance: np.ndarray = np.nan_to_num(
            np.stack(
                [
                    (
                        clearsky_total_diffuse_surface_irradiance
                        * (
                            ground_to_surface_projection_frame.iloc[
                                ground_index
                            ].to_numpy()
                            * polytunnel.transmissivity
                        )[np.newaxis, :, np.newaxis]
                    ).sum(axis=1)
                    for ground_index, _ in tqdm(
                        enumerate(polytunnel.ground_mesh),
                        desc="Diffuse day ground-irradiance calculation",
                        leave=False,
                        total=len(polytunnel.ground_mesh),
                    )
                ],
                axis=1,
            )
        )

        # Compute the diffuse irradiance on the ground given diffuse and direct
        # assumptions for the Hadlow data

        # The code here takes the diffuse irradiance on the surface, which hascolumns that
        # run as time, index, and wavelength. The ground_to_surface_projection_frame is
        # multiplied along the second axis.
        diffuse_day_ground_diffuse_irradiance: np.ndarray = np.stack(
            [
                (
                    diffuse_day_total_diffuse_surface_irradiance
                    * (
                        ground_to_surface_projection_frame.iloc[ground_index].to_numpy()
                        * polytunnel.transmissivity
                    )[np.newaxis, :, np.newaxis]
                ).sum(axis=1)
                for ground_index, _ in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Diffuse day ground-irradiance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            ],
            axis=1,
        )

        direct_day_ground_diffuse_irradiance: np.ndarray = np.stack(
            [
                (
                    direct_day_total_diffuse_surface_irradiance
                    * (
                        ground_to_surface_projection_frame.iloc[ground_index].to_numpy()
                        * polytunnel.transmissivity
                    )[np.newaxis, :, np.newaxis]
                ).sum(axis=1)
                for ground_index, _ in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Diffuse day ground-irradiance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            ],
            axis=1,
        )

        # TODO: Implement diffuse light from the ends of the polytunnel.

    # Readjust the direct day to compute a direct-beam irradiance.
    direct_day_ground_direct_beam_irradiance: np.ndarray = (
        direct_day_ground_direct_irradiance
        / np.array([cos(position.theta_spherical) for position in solar_positions])[
            :, np.newaxis, np.newaxis
        ]
    )

    # Compute the total on-the-ground irradiance map
    with time_execution("Global on-the-ground calculation"):
        # Compute a generalised on-the-ground map for clearsky conditions.
        clearsky_total_ground_irradiance_map: np.ndarray = (
            clearsky_ground_direct_irradiance_map + clearsky_ground_diffuse_irradiance
        )

        # Store a generalised cloudy-day on-the-ground map where all irradiance is
        # cloudy (in terms of its spectrum) and diffuse in its nature.
        cloudysky_total_ground_irradiance_map: np.ndarray = (
            diffuse_day_ground_diffuse_irradiance.copy()
        )

        # Compute a map where all irradiance on the surface is taken to be direct
        # solar irradiance but is scaled by the Hadlow numbers.
        direct_day_total_ground_irradiance_map: np.ndarray = (
            direct_day_ground_diffuse_irradiance + direct_day_ground_direct_irradiance
        )

        #######################
        # Plotting code No. 1 #
        #######################

        ########################
        # Plotting code No. 1a #
        ########################

    # Reset missing indices
    try:
        clearsky_total_ground_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
        clearsky_ground_direct_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
        clearsky_ground_diffuse_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )

        direct_day_ground_diffuse_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
        direct_day_ground_direct_irradiance.index = (
            dni_to_weather_adjustment_factor.index
        )
        direct_day_ground_direct_beam_irradiance.index = (
            dni_to_weather_adjustment_factor.index
        )
        direct_day_total_ground_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
        direct_day_total_ground_with_beam_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )

        diffuse_day_ground_diffuse_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
        diffuse_day_total_ground_irradiance_map.index = (
            dni_to_weather_adjustment_factor.index
        )
    except (AttributeError, NameError):
        pass

    # Save output files for on-the-ground information:
    # Save the maps of PAR on the ground for each hour.
    clearsky_par = 10000 * integrate_spectrum(
        power_spectrum_to_par_spectrum(
            clearsky_total_ground_irradiance_map, PAR_WAVELENGTH_RANGE, wavelength_range
        ),
        wavelength_step_nm,
    )
    with open(
        f"clearsky_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (clearsky_frame := pd.DataFrame(clearsky_par)).to_csv(csv_file)

    direct_sky_par = 10000 * integrate_spectrum(
        power_spectrum_to_par_spectrum(
            direct_day_total_ground_irradiance_map,
            PAR_WAVELENGTH_RANGE,
            wavelength_range,
        ),
        wavelength_step_nm,
    )
    with open(
        f"direct_sky_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (direct_frame := pd.DataFrame(direct_sky_par)).to_csv(csv_file)

    cloudy_sky_par = 10000 * integrate_spectrum(
        power_spectrum_to_par_spectrum(
            cloudysky_total_ground_irradiance_map,
            PAR_WAVELENGTH_RANGE,
            wavelength_range,
        ),
        wavelength_step_nm,
    )
    with open(
        f"cloudy_sky_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (cloudy_frame := pd.DataFrame(cloudy_sky_par)).to_csv(csv_file)

    # Save the mean PAR and stddev in the PAR across the ground at each hour.
    with open(
        f"clearsky_mean_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        clearsky_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"clearsky_stddev_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        clearsky_frame.std(axis=1).to_csv(csv_file)

    with open(
        f"direct_sky_mean_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        direct_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"direct_sky_stddev_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        direct_frame.std(axis=1).to_csv(csv_file)

    with open(
        f"cloudy_sky_mean_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        cloudy_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"cloudy_sky_stddev_par_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        cloudy_frame.std(axis=1).to_csv(csv_file)

    # Save the photon flux on the ground at each hour.
    clearsky_flux = 10000 * integrate_spectrum(
        power_spectrum_to_flux_spectrum(
            clearsky_total_ground_irradiance_map, wavelength_range
        ),
        wavelength_step_nm,
    )
    with open(
        f"clearsky_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (clearsky_frame := pd.DataFrame(clearsky_flux)).to_csv(csv_file)

    direct_sky_flux = 10000 * integrate_spectrum(
        power_spectrum_to_flux_spectrum(
            direct_day_total_ground_irradiance_map, wavelength_range
        ),
        wavelength_step_nm,
    )
    with open(
        f"direct_sky_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (direct_frame := pd.DataFrame(direct_sky_flux)).to_csv(csv_file)

    cloudy_sky_flux = 10000 * integrate_spectrum(
        power_spectrum_to_flux_spectrum(
            cloudysky_total_ground_irradiance_map, wavelength_range
        ),
        wavelength_step_nm,
    )
    with open(
        f"cloudy_sky_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        (cloudy_frame := pd.DataFrame(cloudy_sky_flux)).to_csv(csv_file)

    # Save the mean PAR and stddev in the PAR across the ground at each hour.
    with open(
        f"clearsky_mean_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        clearsky_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"clearsky_stddev_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        clearsky_frame.std(axis=1).to_csv(csv_file)

    with open(
        f"direct_sky_mean_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        direct_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"direct_sky_stddev_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        direct_frame.std(axis=1).to_csv(csv_file)

    with open(
        f"cloudy_sky_mean_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        cloudy_frame.mean(axis=1).to_csv(csv_file)

    with open(
        f"cloudy_sky_stddev_flux_umol_m2_{polytunnel.name}_"
        f"{simulation_start_datetime.strftime("%Y_%m_%d_%H_%M")}_to_"
        f"{simulation_end_datetime.strftime("%Y_%m_%d_%H_%M")}_{INDEX}.csv",
        "w",
        encoding="UTF-8",
    ) as csv_file:
        cloudy_frame.std(axis=1).to_csv(csv_file)

    # Skip and stop if no plots to plot.
    if parsed_args.skip_plots:
        return

    ###########################################################
    # Note: All following code is for plotting purposes only. #
    ###########################################################

    # Create and save plots if requested.
    if not parsed_args.skip_animations:
        with tqdm(desc="Plotting animations", total=8, leave=True) as pbar:
            plot_animation(
                direct_day_ground_diffuse_irradiance,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_direct_day_ground_diffuse_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                direct_day_ground_direct_irradiance,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_direct_day_ground_direct_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                diffuse_day_ground_diffuse_irradiance,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_diffuse_day_ground_diffuse_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                clearsky_ground_diffuse_irradiance,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_clearsky_ground_diffuse_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                clearsky_ground_direct_irradiance_map,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_clearsky_ground_direct_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                clearsky_total_ground_irradiance_map,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_clearsky_total_ground_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                direct_day_total_ground_irradiance_map,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_direct_day_total_ground_irradiance_{polytunnel.name}",
            )
            pbar.update(1)
            plot_animation(
                cloudysky_total_ground_irradiance_map,
                polytunnel,
                wavelength_range,
                index=INDEX,
                modelling_temporal_resolution=parsed_args.modelling_temporal_resolution,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                title=f"par_cloudysky_total_ground_irradiance_map_{polytunnel.name}",
            )
            pbar.update(1)

    with tqdm(desc="Plotting spectra", total=7, leave=True) as pbar:
        # Plot the pyranometer response
        plot_spectrum(
            [
                (
                    adjusted_global_spectrum,
                    f"Global response ({sum(adjusted_global_spectrum):.4g}×)",
                    5,
                ),
                (
                    adjusted_direct_spectrum,
                    f"Direct response ({sum(adjusted_direct_spectrum):.4g}×)",
                    4,
                ),
                (
                    adjusted_diffuse_spectrum,
                    f"Diffuse response ({sum(adjusted_diffuse_spectrum):.4g}×)",
                    2,
                ),
                (
                    adjusted_cloudy_spectrum,
                    f"Cloudy-day response ({sum(adjusted_cloudy_spectrum):.4g}×)",
                    0,
                ),
            ],
            pyranometer_wavelength_range,
            index=INDEX,
            show=False,
            title="pyranometer_response",
        )
        pbar.update(1)

        plot_spectrum(
            [
                (
                    adjusted_global_spectrum,
                    f"Global response ({sum(adjusted_global_spectrum):.4g}×)",
                    5,
                ),
                (
                    adjusted_direct_spectrum,
                    f"Direct response ({sum(adjusted_direct_spectrum):.4g}×)",
                    4,
                ),
                (
                    adjusted_diffuse_spectrum,
                    f"Diffuse response ({sum(adjusted_diffuse_spectrum):.4g}×)",
                    2,
                ),
                (
                    adjusted_cloudy_spectrum,
                    f"Cloudy-day response ({sum(adjusted_cloudy_spectrum):.4g}×)",
                    0,
                ),
            ],
            pyranometer_wavelength_range,
            index=INDEX,
            plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
            spectral_units=SpectralUnits.PAR_FLUX,
            title="pyranometer_response_par_flux",
        )
        pbar.update(1)

        # Plot the received irradiance and photon flux on the ground:
        # - Plot the direct irradiance which was received;
        # - Plot the diffuse irradiance which was received.

        # Construct a palette based on the number of non-zero illuminated ground squares
        _hour: int = 12
        num_grid_indices: int = 0
        for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
            if (
                clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
                > 0
            ):
                num_grid_indices += 1

        import seaborn as sns

        _palette = sns.cubehelix_palette(
            start=0.4, rot=-1.2, n_colors=num_grid_indices, reverse=True
        )

        # Only attempt to plot if non-zero light was received at this hour.
        if len(_palette) > 0:
            plot_spectrum(
                [
                    (
                        clearsky_ground_direct_irradiance_map_with_pv_module_and_spectra[
                            _hour
                        ][
                            grid_index
                        ],
                        f"{grid_index}",
                    )
                    for grid_index in range(
                        len(clearsky_ground_direct_irradiance_map[_hour])
                    )
                    if clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][
                        _hour
                    ]
                    > 0
                ],
                wavelength_range,
                index=INDEX,
                palette=_palette,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                small=False,
                spectral_units=SpectralUnits.PAR_FLUX,
                title="par_on_the_ground_spectra_v1",
                unique_legend=True,
            )
            pbar.update(1)

            # Plot the spectra using a different method.
            _palette = sns.color_palette(
                list(
                    reversed(
                        [
                            "#423252",
                            "#4A688B",
                            "#779FB1",
                            "#36C7B8",
                            "#FBC412",
                            "#FE8224",
                            "#E03944",
                        ]
                    )
                )
            )
            _hour = 12
            plotting_data: list[tuple[np.ndarray | pd.Series, str, float]] = []
            for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
                if (
                    clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][
                        _hour
                    ]
                    > 0
                ):
                    plotting_data.append(
                        (
                            clearsky_ground_direct_irradiance_map[_hour][grid_index],
                            "Through-PV",
                            5,
                            "--",
                        )
                    )
                elif sum(clearsky_ground_direct_irradiance_map[_hour][grid_index]) == 0:
                    plotting_data.append(
                        (
                            clearsky_ground_direct_irradiance_map[_hour][grid_index],
                            "No direct sunlight",
                            6,
                            ":",
                        )
                    )
                elif sum(end_direct_irradiance_map[_hour][grid_index]) > 0:
                    plotting_data.append(
                        (
                            clearsky_ground_direct_irradiance_map[_hour][grid_index],
                            "Through-open ends",
                            1,
                            "-.",
                        )
                    )
                else:
                    plotting_data.append(
                        (
                            clearsky_ground_direct_irradiance_map[_hour][grid_index],
                            "Through-polytunnel",
                            3,
                            "-",
                        )
                    )

            plot_spectrum(
                plotting_data,
                wavelength_range,
                index=INDEX,
                palette=_palette,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                right_axis_data=[
                    (
                        pyranometer_adjusted_interpolated_spectra.direct.values,
                        "Direct-irradiance response",
                        2,
                        "-.",
                    )
                ],
                show=False,
                small=False,
                spectral_units=SpectralUnits.PAR_FLUX,
                title="par_on_the_ground_spectra_v2_with_reference",
                unique_legend=True,
            )
            pbar.update(1)

            plot_spectrum(
                plotting_data,
                wavelength_range,
                index=INDEX,
                palette=_palette,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                right_axis_data=[
                    (
                        pyranometer_adjusted_interpolated_spectra.direct.values,
                        "Direct-irradiance response",
                        2,
                        "-.",
                    )
                ],
                show=False,
                small=True,
                spectral_units=SpectralUnits.PAR_FLUX,
                title="par_on_the_ground_spectra_v2_with_reference_small",
                unique_legend=True,
            )
            pbar.update(1)

            plot_spectrum(
                plotting_data,
                wavelength_range,
                index=INDEX,
                palette=_palette,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                small=False,
                spectral_units=SpectralUnits.PAR_FLUX,
                title="par_on_the_ground_spectra_v2_sans_reference",
                unique_legend=True,
            )
            pbar.update(1)

            plot_spectrum(
                plotting_data,
                wavelength_range,
                index=INDEX,
                palette=_palette,
                plotting_wavelength_range=PAR_WAVELENGTH_RANGE,
                show=False,
                small=True,
                spectral_units=SpectralUnits.PAR_FLUX,
                title="par_on_the_ground_spectra_v2_sans_reference_small",
                unique_legend=True,
            )
            pbar.update(1)
        else:
            pbar.update(5)

        #######################
        # Plotting code No. 4 #
        #######################

        if not isinstance(stack_tmm, DummyTMM):
            from matplotlib import colors as m_colors

            # Plot the spectra at a specific hour for each element within the ground mesh.
            # NOTE: Colours indicate the index of the element providing surface irradiation.

            # Determine the number of non-zero elements
            _hour: int = 12
            num_grid_indices: int = 0
            for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
                if (
                    clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][
                        _hour
                    ]
                    > 0
                ):
                    num_grid_indices += 1

            sns.set_palette(
                sns.cubehelix_palette(
                    start=0.4, rot=-1.2, n_colors=num_grid_indices, reverse=True
                )
            )
            sns.set_palette("viridis", n_colors=num_grid_indices)

            plt.figure(figsize=(171 * MM, 120 * MM))
            dashes = Dashes()
            _zorder = 0
            grid_indices: list[int] = []
            for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
                if (
                    clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][
                        _hour
                    ]
                    > 0
                ):
                    _color = f"C{grid_index}"
                    _zorder += 1
                    grid_indices.append(grid_index)
                else:
                    _color = "C0"
                plt.plot(
                    wavelength_range,
                    clearsky_ground_direct_irradiance_map_with_pv_module_and_spectra[
                        _hour
                    ][grid_index],
                    dashes=next(dashes),
                    label=f"#{_color}" if _color != "C0" else None,
                    color=_color,
                    zorder=0 if _color == "C0" else _zorder,
                )

            plt.legend().remove()

            norm = plt.Normalize(
                -0.5,
                clearsky_ground_direct_irradiance_map_with_pv_module.shape[1] + 0.5,
            )
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=m_colors.LinearSegmentedColormap.from_list(
                    "Custom",
                    sns.color_palette().as_hex(),
                    num_grid_indices,
                ),
                norm=norm,
            )

            colorbar = (axis := plt.gca()).figure.colorbar(
                scalar_mappable,
                ax=axis,
                label="Surface index illuminating",
                pad=(_pad := 0.125),
            )
            colorbar.set_ticks(grid_indices)
            colorbar.set_ticklabels(
                [
                    entry if index % 3 == 0 else None
                    for index, entry in enumerate(grid_indices)
                ]
            )

            axis.tick_params(axis="both", which="major", labelsize=7)
            plt.xlabel("Wavelength / nm", fontsize=7)
            plt.ylabel("Irradiance / W/m$^2$nm", fontsize=7)

            (right_axis := axis.twinx()).plot(
                wavelength_range,
                pyranometer_adjusted_interpolated_spectra.direct.values,
                "--",
                color="C9",
            )
            right_axis.set_ylabel("Direct-irradiance response / normalised units")
            right_axis.tick_params(axis="both", which="major", labelsize=7)

            plt.savefig(
                f"ground_through_pv_spectra_profiles_{_hour}_{INDEX}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )
            plt.savefig(
                f"ground_through_pv_spectra_profiles_{_hour}_{INDEX}.png",
                format="png",
                bbox_inches="tight",
                pad_inches=0.05,
                transparent=True,
                dpi=1200,
            )
            pbar.update(1)

    if alternative_weather_data is not None:
        diffusivity_series = (
            alternative_weather_data[INCOMING_SHORTWAVE_DIFFUSE]
            / (
                alternative_weather_data[INCOMING_SHORTWAVE_DIFFUSE]
                + alternative_weather_data[INCOMING_SHORTWAVE_DIRECT]
            )
        )[hadlow_dni_slice.index]

        if parsed_args.weather_as_diffusivity_only:
            # If the weather data should be used only to compute the
            # diffusivity, then use the weather data file to predict the
            # weather based on the existing basis values and the diffusivity
            # provided.

            # Compute the on-the-ground irradiance based on the diffusivity
            # fraction.
            predicted_day_ground_diffuse_map: np.ndarray = (
                cloudysky_total_ground_irradiance_map
                * diffusivity_series.to_numpy()[:, None, None]
                + direct_day_ground_diffuse_irradiance
                * (1 - diffusivity_series).to_numpy()[:, None, None]
            )

            predicted_day_ground_direct_beam_map: np.ndarray = (
                direct_day_ground_direct_beam_irradiance
                * (1 - diffusivity_series).to_numpy()[:, None, None]
            )

            predicted_day_ground_total_map: np.ndarray = (
                predicted_day_ground_direct_beam_map + predicted_day_ground_diffuse_map
            )

        else:
            predicted_day_ground_diffuse_map = (
                cloudysky_total_ground_irradiance_map
                + direct_day_ground_diffuse_irradiance
            )

            predicted_day_ground_direct_beam_map = (
                direct_day_ground_direct_beam_irradiance
            )

            predicted_day_ground_total_map = (
                predicted_day_ground_direct_beam_map
                + predicted_day_ground_direct_beam_map
            )

    # Parse the validation data if provided to compare against.
    if parsed_args.validation_filename is not None:
        with time_execution("Generating validation plots"):
            with tqdm(
                desc="Generating validation plots",
                leave=False,
                total=7 + 3 * (alternative_weather_data is not None),
            ) as pbar:
                try:
                    with open(
                        parsed_args.validation_filename, "r", encoding="UTF-8"
                    ) as validation_file:
                        validation_data: pd.DataFrame = pd.read_csv(
                            validation_file, header=0, index_col=0
                        )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not find validation file: {parsed_args.validation_filename}"
                    ) from None

                if parsed_args.validation_index is None:
                    raise Exception(
                        "Must specify validation index if carrying out a validation."
                    )

                # Parse out the section of the validation data which is relevant.
                validation_data.index = pd.Index(
                    [
                        datetime.datetime.strptime(entry, "%m/%d/%y %H:%M")
                        for entry in validation_data.index
                    ]
                )
                validation_data[ValidationColumns.DIFFUSE_ERROR.value] = (
                    DIFFUSE_PAR_ERROR
                    * validation_data[ValidationColumns.DIFFUSE_PAR.value]
                )
                validation_data[ValidationColumns.TOTAL_ERROR.value] = (
                    TOTAL_PAR_ERROR * validation_data[ValidationColumns.TOTAL_PAR.value]
                )
                validation_data[ValidationColumns.DIRECT_ERROR.value] = [
                    sqrt(entry)
                    for entry in (
                        validation_data[ValidationColumns.DIFFUSE_ERROR.value]
                        / validation_data[ValidationColumns.DIFFUSE_PAR.value]
                        + validation_data[ValidationColumns.TOTAL_ERROR.value]
                        / validation_data[ValidationColumns.TOTAL_PAR.value]
                    )
                ]

                dir_day_gnd_tot_val: pd.DataFrame = pd.merge(
                    pd.DataFrame(
                        integrate_spectrum(
                            direct_day_total_ground_irradiance_map, wavelength_step_nm
                        ),
                        index=hadlow_dni_slice.index,
                    ),
                    validation_data,
                    left_index=True,
                    right_index=True,
                )
                dir_day_gnd_dir_val: pd.DataFrame = pd.merge(
                    pd.DataFrame(
                        integrate_spectrum(
                            direct_day_ground_direct_beam_irradiance, wavelength_step_nm
                        ),
                        index=hadlow_dni_slice.index,
                    ),
                    validation_data,
                    left_index=True,
                    right_index=True,
                )
                dir_day_gnd_dif_val: pd.DataFrame = pd.merge(
                    pd.DataFrame(
                        integrate_spectrum(
                            direct_day_ground_diffuse_irradiance, wavelength_step_nm
                        ),
                        index=hadlow_dni_slice.index,
                    ),
                    validation_data,
                    left_index=True,
                    right_index=True,
                )

                dif_day_gnd_tot_val: pd.DataFrame = pd.merge(
                    pd.DataFrame(
                        integrate_spectrum(
                            cloudysky_total_ground_irradiance_map, wavelength_step_nm
                        ),
                        index=hadlow_dni_slice.index,
                    ),
                    validation_data,
                    left_index=True,
                    right_index=True,
                )

                try:
                    sns.set_palette(
                        # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
                        [
                            "#423252",
                            "#4A688B",
                            "#779FB1",
                            "#36C7B8",
                            "#FBC412",
                            "#e04606",
                        ],
                    )
                except UnboundLocalError:
                    import seaborn as sns

                    sns.set_palette(
                        # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
                        [
                            "#423252",
                            "#4A688B",
                            "#779FB1",
                            "#36C7B8",
                            "#FBC412",
                            "#e04606",
                        ],
                    )

                try:
                    plt.close()

                except UnboundLocalError:
                    import matplotlib.pyplot as plt

                #######################
                # Plotting code No. 1 #
                #######################

                # import matplotlib.pyplot as plt
                # import matplotlib.animation as animation
                # import seaborn as sns
                # import numpy as np

                # fig, ax = plt.subplots(figsize=(171*MM, 120*MM))

                # # Create initial heatmap with dummy data
                # initial_data = np.reshape(
                #     diffuse_day_total_ground_irradiance_map.iloc[0],
                #     (
                #         _dim_x := polytunnel.meshgrid_resolution,
                #         _dim_y := polytunnel.length_wise_meshgrid_resolution,
                #     ),
                # )
                # vmin = 0
                # vmax = max(diffuse_day_total_ground_irradiance_map.max(axis=0))
                # heatmap = sns.heatmap(
                #     initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax
                # )

                # _ten_minutes: int = int(
                #     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
                # )

                # def update(time_index: int):
                #     ax.clear()  # clear previous heatmap
                #     data = np.reshape(
                #         diffuse_day_total_ground_irradiance_map.iloc[time_index], (_dim_x, _dim_y)
                #     )
                #     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
                #     ax.set_title(
                #         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
                #     )

                # # Create the animation
                # ani = animation.FuncAnimation(
                #     fig,
                #     update,
                #     frames=len(diffuse_day_total_ground_irradiance_map),
                #     interval=300,
                #     repeat=False,
                # )
                # ani.save("diffuse_day_total_ground_irradiance_map_2.gif", writer="pillow", fps=5)
                # plt.show()

                #######################
                # Plotting code No. 3 #
                #######################

                dif_day_gnd_tot_val.index = pd.Index(
                    [entry.time().strftime("%H") for entry in dif_day_gnd_tot_val.index]
                )
                dir_day_gnd_tot_val.index = pd.Index(
                    [entry.time().strftime("%H") for entry in dir_day_gnd_tot_val.index]
                )
                dir_day_gnd_dir_val.index = pd.Index(
                    [entry.time().strftime("%H") for entry in dir_day_gnd_dir_val.index]
                )
                dir_day_gnd_dif_val.index = pd.Index(
                    [entry.time().strftime("%H") for entry in dir_day_gnd_dif_val.index]
                )
                diffusivity_series.index = pd.Index(
                    [entry.time().strftime("%H") for entry in diffusivity_series.index]
                )

                plt.figure(
                    figsize=(83 * MM, 60 * MM)
                )  # 171 * MM, 120 * MM  # 83 * MM, 60 * MM
                sns.scatterplot(
                    x=dir_day_gnd_tot_val.index,
                    y=dir_day_gnd_tot_val[parsed_args.validation_index],
                    color="C4",
                    label="Direct-day total prediction",
                    marker="h",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_tot_val.index,
                    dir_day_gnd_tot_val[parsed_args.validation_index],
                    color="C4",
                )
                sns.scatterplot(
                    x=dif_day_gnd_tot_val.index,
                    y=dif_day_gnd_tot_val[parsed_args.validation_index],
                    color="C3",
                    label="Diffuse-day total prediction",
                    marker="h",
                    s=40,
                )
                plt.plot(
                    dif_day_gnd_tot_val.index,
                    dif_day_gnd_tot_val[parsed_args.validation_index],
                    color="C3",
                )
                sns.scatterplot(
                    x=dir_day_gnd_tot_val.index,
                    y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    color="C0",
                    label="Total PAR",
                    marker="H",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_tot_val.index,
                    dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    color="C0",
                )
                plt.errorbar(
                    dir_day_gnd_dir_val.index,
                    dir_day_gnd_dir_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C0",
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                axis_right = (axis_left := plt.gca()).twinx()
                axis_right.set_ylabel("Diffusivity")
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                sns.scatterplot(
                    x=dir_day_gnd_tot_val.index,
                    y=dir_day_gnd_tot_val["diffusivity"],
                    alpha=0.7,
                    color="C1",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)

                plt.savefig(
                    f"validation_{parsed_args.validation_index}_{alt_weather}total_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 4 #
                #######################

                plt.figure(figsize=(83 * MM, 60 * MM))
                sns.scatterplot(
                    x=dir_day_gnd_dir_val.index,
                    y=dir_day_gnd_dir_val[parsed_args.validation_index],
                    color="C4",
                    label="Direct-day direct prediction",
                    marker="h",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_dir_val.index,
                    dir_day_gnd_dir_val[parsed_args.validation_index],
                    color="C4",
                )
                sns.scatterplot(
                    x=dir_day_gnd_dir_val.index,
                    y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    color="C1",
                    label="Direct PAR",
                    marker="H",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_dir_val.index,
                    dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    color="C1",
                )
                plt.errorbar(
                    dir_day_gnd_dir_val.index,
                    dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.DIRECT_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C1",
                )

                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                axis_right = (axis_left := plt.gca()).twinx()
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                sns.scatterplot(
                    x=dir_day_gnd_tot_val.index,
                    y=dir_day_gnd_tot_val["diffusivity"],
                    alpha=0.7,
                    color="C2",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()
                axis_right.set_ylabel("Diffusivity")

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)
                plt.savefig(
                    f"validation_{parsed_args.validation_index}_{alt_weather}direct_diff_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_"
                    f"{parsed_args.start_time.replace(':','_')}_{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 5 #
                #######################

                plt.figure(figsize=(83 * MM, 60 * MM))
                sns.scatterplot(
                    x=dir_day_gnd_dif_val.index,
                    y=dir_day_gnd_dif_val[parsed_args.validation_index],
                    color="C4",
                    label="Direct-day prediction",
                    marker="h",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_dif_val.index,
                    dir_day_gnd_dif_val[parsed_args.validation_index],
                    color="C4",
                )
                sns.scatterplot(
                    x=dif_day_gnd_tot_val.index,
                    y=dif_day_gnd_tot_val[parsed_args.validation_index].values,
                    color="C3",
                    label="Diffuse-day prediction",
                    marker="h",
                    s=40,
                )
                plt.plot(
                    dif_day_gnd_tot_val.index,
                    dif_day_gnd_tot_val[parsed_args.validation_index],
                    color="C3",
                )
                sns.scatterplot(
                    x=dir_day_gnd_dif_val.index,
                    y=dir_day_gnd_dif_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    color="C1",
                    label="Diffuse PAR",
                    marker="H",
                    s=40,
                )
                plt.plot(
                    dir_day_gnd_dif_val.index,
                    dir_day_gnd_dif_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    color="C1",
                )
                plt.errorbar(
                    dir_day_gnd_dir_val.index,
                    dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C1",
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                plt.legend(fontsize=7)

                axis_right = (axis_left := plt.gca()).twinx()
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                axis_right.set_ylabel("Diffusivity")
                sns.scatterplot(
                    x=dir_day_gnd_tot_val.index,
                    y=dir_day_gnd_tot_val["diffusivity"],
                    alpha=0.7,
                    color="C2",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)
                plt.savefig(
                    f"validation_{parsed_args.validation_index}_{alt_weather}diffuse_diff_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 6 #
                #######################

                plt.figure(figsize=(83 * MM, 60 * MM))
                sns.boxplot(
                    dif_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
                    boxprops=dict(alpha=0.75),
                    color="C3",
                    label="Diffuse-day prediction",
                    saturation=1,
                    # linecolor="C3",
                    zorder=0,
                )
                sns.boxplot(
                    dir_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
                    boxprops=dict(alpha=0.75),
                    color="C4",
                    label="Direct-day prediction",
                    # linecolor="C4",
                    saturation=1,
                    zorder=0,
                )
                sns.scatterplot(
                    x=range(len(dir_day_gnd_tot_val)),
                    y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    color="C0",
                    label="Total PAR",
                    marker="H",
                    s=60,
                    zorder=1,
                )
                plt.plot(
                    range(len(dir_day_gnd_tot_val)),
                    dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    color="C0",
                    zorder=1,
                )
                plt.errorbar(
                    x=range(len(dir_day_gnd_tot_val)),
                    y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C0",
                    zorder=1,
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                axis_right = (axis_left := plt.gca()).twinx()
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                axis_right.set_ylabel("Diffusivity")
                sns.scatterplot(
                    x=range(len(dir_day_gnd_tot_val)),
                    y=dir_day_gnd_tot_val["diffusivity"],
                    alpha=0.7,
                    color="C1",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                    zorder=1,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)

                plt.xticks(
                    plt.xticks()[0][::3],
                    [entry for entry in dir_day_gnd_tot_val.index][::3],
                )

                plt.savefig(
                    "validation_total_map_boxplot_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 7 #
                #######################

                plt.figure(figsize=(83 * MM, 60 * MM))
                sns.boxplot(
                    dif_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
                    boxprops=dict(alpha=0.75),
                    color="C3",
                    label="Diffuse-day prediction",
                    saturation=1,
                    # linecolor="C3",
                    zorder=0,
                )
                sns.boxplot(
                    dir_day_gnd_dif_val.reset_index(drop=True).transpose()[:-13],
                    boxprops=dict(alpha=0.75),
                    color="C4",
                    label="Direct-day prediction",
                    # linecolor="C4",
                    saturation=1,
                    zorder=0,
                )
                sns.scatterplot(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    color="C1",
                    label="Diffuse PAR",
                    marker="H",
                    s=60,
                    zorder=1,
                )
                plt.plot(
                    range(len(dir_day_gnd_dir_val)),
                    dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    color="C1",
                    zorder=1,
                )
                plt.errorbar(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C1",
                    zorder=1,
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                axis_right = (axis_left := plt.gca()).twinx()
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                sns.scatterplot(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val["diffusivity"],
                    alpha=0.7,
                    color="C1",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                    zorder=1,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()
                axis_right.set_ylabel("Diffusivity")

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)

                plt.xticks(
                    plt.xticks()[0][::3],
                    [entry for entry in dir_day_gnd_dir_val.index][::3],
                )

                plt.savefig(
                    "validation_diffuse_map_boxplot_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 8 #
                #######################

                plt.figure(figsize=(83 * MM, 60 * MM))
                sns.scatterplot(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val.reset_index(drop=True)
                    .transpose()[:-13]
                    .mean(axis=0),
                    # boxprops=dict(alpha=0.75),
                    color="C4",
                    label="Direct-day prediction",
                    # linecolor="C4",
                    marker="h",
                    s=60,
                    # saturation=1,
                    zorder=0,
                )
                plt.plot(
                    range(len(dir_day_gnd_dir_val)),
                    dir_day_gnd_dir_val.reset_index(drop=True)
                    .transpose()[:-13]
                    .mean(axis=0),
                    color="C4",
                    zorder=0,
                )
                sns.scatterplot(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    color="C1",
                    label="Direct PAR",
                    marker="H",
                    s=60,
                    zorder=1,
                )
                plt.plot(
                    range(len(dir_day_gnd_dir_val)),
                    dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    color="C1",
                    zorder=1,
                )
                plt.errorbar(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.DIRECT_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C1",
                    zorder=1,
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                axis_right = (axis_left := plt.gca()).twinx()
                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)
                axis_right.set_ylabel("Diffusivity")
                sns.scatterplot(
                    x=range(len(dir_day_gnd_dir_val)),
                    y=dir_day_gnd_dir_val["diffusivity"],
                    alpha=0.7,
                    color="C1",
                    label="Diffusivity",
                    marker="D",
                    s=40,
                    zorder=1,
                )
                left_handles, left_labels = axis_left.get_legend_handles_labels()
                axis_left.legend().remove()
                right_handles, right_labels = axis_right.get_legend_handles_labels()
                axis_right.legend().remove()

                plt.legend(
                    left_handles + right_handles,
                    left_labels + right_labels,
                    loc="upper right",
                    fontsize=7,
                )
                axis_right.set_ylim(-0.05, 1.05)
                axis_left.set_ylim(-25, 825)

                plt.xticks(
                    list(range(len(dir_day_gnd_dir_val.index)))[::3],
                    [entry for entry in dir_day_gnd_dir_val.index][::3],
                )

                plt.savefig(
                    "validation_direct_map_boxplot_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

                #######################
                # Plotting code No. 9 #
                #######################

                # Compute the cloudiness based on the on-the-ground PAR seen.
                diffusivity: pd.Series = (
                    dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48
                    - dir_day_gnd_tot_val[parsed_args.validation_index]
                ) / (
                    dif_day_gnd_tot_val[parsed_args.validation_index]
                    - dir_day_gnd_tot_val[parsed_args.validation_index]
                )

                diffusivity_error = abs(diffusivity * 0.1)

                plt.figure(figsize=(83 * MM, 60 * MM))
                axis_right = (axis_left := plt.gca()).twinx()

                sns.scatterplot(
                    x=range(len(dir_day_gnd_tot_val)),
                    y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    ax=axis_left,
                    color="C0",
                    label="Total PAR",
                    marker="H",
                    s=60,
                    zorder=1,
                )
                axis_left.plot(
                    range(len(dir_day_gnd_tot_val)),
                    dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    color="C0",
                    zorder=1,
                )
                axis_left.errorbar(
                    x=range(len(dir_day_gnd_tot_val)),
                    y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
                    yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value]
                    * 0.48,
                    ls="none",
                    color="C0",
                    zorder=1,
                )
                plt.xlabel("Time / h")
                plt.ylabel("Irradiance / W/m$^2$")

                sns.scatterplot(
                    x=range(len(diffusivity)),
                    y=diffusivity,
                    alpha=0.7,
                    ax=axis_right,
                    color="C2",
                    label="Predicted weather diffusivity",
                    marker="X",
                    s=40,
                    zorder=1,
                )
                axis_right.errorbar(
                    x=(x_range := range(len(dir_day_gnd_dir_val))),
                    y=diffusivity,
                    yerr=diffusivity_error,
                    ls="none",
                    color="C2",
                    zorder=1,
                )
                axis_right.set_xlabel("Time / h")
                axis_left.set_ylabel("Irradiance / W/m$^2$")
                axis_right.set_ylabel("Diffusivity")

                plt.xticks(
                    list(range(len(dir_day_gnd_dir_val.index)))[::4],
                    [entry for entry in dir_day_gnd_dir_val.index][::4],
                )

                lower_ylim: float = -0.75
                upper_ylim: float = 4.75
                axis_right.fill_between(
                    x_range,
                    [lower_ylim] * len(x_range),
                    [0] * len(x_range),
                    alpha=0.3,
                    color="grey",
                    hatch="//",
                    zorder=0,
                    label="Out-of-bounds result",
                )
                axis_right.fill_between(
                    x_range,
                    [1] * len(x_range),
                    [upper_ylim] * len(x_range),
                    alpha=0.3,
                    color="grey",
                    hatch="//",
                    zorder=0,
                )

                axis_left.set_ylim(-25, 825)
                axis_right.set_ylim(lower_ylim, upper_ylim)

                handles_l, labels_l = axis_left.get_legend_handles_labels()
                handles_r, labels_r = axis_right.get_legend_handles_labels()

                axis_left.tick_params(axis="both", which="major", labelsize=7)
                axis_right.tick_params(axis="both", which="major", labelsize=7)

                axis_left.legend().remove()
                axis_right.legend().remove()
                axis_right.legend(handles_l + handles_r, labels_l + labels_r)

                plt.savefig(
                    "validation_diffusivity_prediction_"
                    f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
                    f"{parsed_args.start_time.replace(':','_')}_"
                    f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                pbar.update(1)

    # import pdb

    # pdb.set_trace()

    if parsed_args.debug:
        plt.show()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
