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
import datetime
import enum
import os
import re
import sys
import time

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, cos, floor, pi
from typing import Any, Callable, Generator, Match, Pattern

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
import yaml

from matplotlib import rc, rcParams
from numpy import inf
from tqdm import tqdm

from src.polytunnel_irradiance_model.__utils__ import *
from src.polytunnel_irradiance_model.functions import *
from src.polytunnel_irradiance_model.polytunnel import (
    calculate_adjacent_polytunnel_shading,
    calculate_and_update_intercept_planes,
    calculate_solid_angles,
    EndType,
    MeshPoint,
    NotInterceptError,
    Polytunnel,
    Plane,
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

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r".*vectorized_tmm_dispersive_multistack.*"
)

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

# MODULES:
#   Keyword for parsing PV-module information.
MODULES: str = "modules"

# MM:
#   Conversion factor from mm to inches.
MM = 1 / 25.4

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

# VERSION_REGEX:
#   Regex used to extract the main version number.
VERSION_REGEX: Pattern[str] = re.compile(r"(?P<number>\d\.\d\.\d)([\.](?P<post>.*))?")


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
        default=51.1950,
        help="The latitude of the location for which weather data should be used.",
    )
    simulation_arguments.add_argument(
        "--longitude",
        "-lon",
        type=float,
        default=0.2757,
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
        "-rtm",
        type=float,
        default=30,
        help="The temporal resolution, in minutes, to use when simulating throughout "
        "the day.",
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

    return parser.parse_args(args)


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
        with open(
            os.path.join(HADLOW_WEATHER_FILENAME), "r", encoding="UTF-8"
        ) as hadlow_weather_file:
            hadlow_weather_data: pd.DataFrame = (
                pd.read_csv(hadlow_weather_file).drop([0, 1]).set_index("parameter-id")
            )

        hadlow_weather_data.index = pd.DatetimeIndex(hadlow_weather_data.index)

        # Slice using the start and end weather values.
        hadlow_dni_slice: pd.DataFrame = hadlow_weather_data[
            parsed_args.start_time.replace("T", " ")
            .replace("Z", "") : parsed_args.end_time.replace("T", " ")
            .replace("Z", "")
        ][INCOMING_SHORTWAVE].astype(float)

        dhi_to_hadlow_adjustment_factor: pd.DataFrame = pd.concat(
            [
                (hadlow_dni_slice / clearsky_irradiance["dhi"])
                .clip(0, None)
                .replace([inf, -inf], 0)
            ]
            * len(polytunnel.surface_mesh),
            axis=1,
        )
        dni_to_hadlow_adjustment_factor: pd.DataFrame = pd.concat(
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

    if not os.path.isfile(
        surface_shaded_map_filename := os.path.join(
            AUTO_GENERATED,
            polytunnel.name,
            f"{polytunnel.name}_surface_shaded_map_"
            f"{parsed_args.start_time.replace(":","_")}_"
            f"{parsed_args.end_time.replace(":","_")}.csv",
        )
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
                            and ((meshpoint._normal_vector * solar_position) > 0)
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

    # Carry out a calculation if the outputs have not already been saved.
    if not os.path.isfile(
        diffuse_surface_filename := os.path.join(
            AUTO_GENERATED,
            polytunnel.name,
            f"{polytunnel.name}_diffuse_surface_irradiance_"
            f"{parsed_args.start_time.replace(":","_")}_"
            f"{parsed_args.end_time.replace(":","_")}.csv",
        )
    ):

        # Determine the intercept lines with neighbouring polytunnels.
        # calculate_and_update_intercept_planes(polytunnel)
        with time_execution("Direct surface calculation"):
            # Compute the solar position dot-product across the surface of the polytunnel.
            dot_product_map = pd.DataFrame(
                {
                    meshpoint_index: [
                        meshpoint._normal_vector * solar_position
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

    diffuse_surface_irradiance.index = hadlow_dni_slice.index
    direct_surface_irradiance.index = hadlow_dni_slice.index

    if not os.path.isfile(
        mesh_mesh_filename := os.path.join(
            AUTO_GENERATED,
            polytunnel.name,
            f"{polytunnel.name}_{polytunnel.meshgrid_resolution}_by_"
            f"{polytunnel.length_wise_meshgrid_resolution}_mesh_mesh_distance.json",
        )
    ):
        with time_execution("Mesh-mesh distance calculation"):
            # Consider each point on the surface as imparting diffuse light on the ground.
            ground_to_surface_projection_map: defaultdict[int, dict[int, float]] = {
                ground_index: {
                    # Angle between vector from ground to surface, dotted with the normal to
                    # the ground;
                    surface_index: abs(
                        (_vector := (ground_meshpoint - surface_meshpoint))
                        * ground_meshpoint._normal_vector
                    )
                    # multiplied by the angle between the ground-to-surface veccto and the
                    # normal of the surface;
                    * abs(_vector * surface_meshpoint._normal_vector)
                    # multiplied by the area of the surface element to go from Watts to
                    # Watts per meter squared;
                    * surface_meshpoint.area
                    # all normalised by the 1/distance^2 to scale back to Watts/meter^2.
                    / (
                        abs(surface_meshpoint._normal_vector)
                        * abs(ground_meshpoint._normal_vector)
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

    with time_execution("Readjusting with Hadlow data"):
        clearsky_total_diffuse_surface_irradiance = (
            diffuse_surface_irradiance.reset_index(drop=True)
            + (
                diffusivity := (
                    parsed_args.diffusivity
                    if parsed_args.diffusivity is not None
                    else polytunnel.diffusivity
                )
            )
            * direct_surface_irradiance.reset_index(drop=True)
        )

        diffuse_day_total_diffuse_surface_irradiance = (
            diffuse_surface_irradiance * dhi_to_hadlow_adjustment_factor
        )
        direct_day_total_diffuse_surface_irradiance = (
            clearsky_total_diffuse_surface_irradiance
            * dni_to_hadlow_adjustment_factor.reset_index(drop=True)
        )

    clearsky_total_diffuse_surface_irradiance.index = hadlow_dni_slice.index
    diffuse_day_total_diffuse_surface_irradiance.index = hadlow_dni_slice.index
    direct_day_total_diffuse_surface_irradiance.index = hadlow_dni_slice.index

    # Calculate the amount of polytunnel surface sunlight which will reach the ground,
    # both as diffuse and direct components.

    # Compute the amount of direct light reaching the ground.
    with time_execution("Direct on-the-ground calculation"):
        clearsky_ground_direct_irradiance_map: pd.DataFrame = pd.DataFrame(
            [
                ground_direct_irradiance(
                    polytunnel.ground_mesh,
                    polytunnel,
                    (
                        surface_shaded_map.loc[time_index]
                        * clearsky_irradiance["dni"].iloc[time_index]
                        * (1 - diffusivity)
                    ).reset_index(drop=True)
                    * polytunnel_surface_pv_uncovered_fraction_mask.iloc[time_index],
                    solar_position,
                    diffusivity=parsed_args.diffusivity,
                )
                for time_index, solar_position in tqdm(
                    enumerate(solar_positions),
                    desc="Direct ground irradiance calculation",
                    leave=False,
                    total=len(solar_positions),
                )
            ]
        )

        #######################
        # Plotting code No. 1 #
        #######################

        # If the ends are open, add the irradiance from the ends.
        if polytunnel.ends == EndType.OPEN:
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
            end_direct_irradiance_map: pd.DataFrame | None = (
                end_intercept_projection.transpose()
                .mul(clearsky_irradiance["dni"].values)
                .transpose()
            )

            clearsky_ground_direct_irradiance_map += end_direct_irradiance_map

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
                end_direct_irradiance_map.to_csv(end_irradiance_file)

            #######################
            # Plotting code No. 1 #
            #######################

        direct_day_ground_direct_irradiance = (
            clearsky_ground_direct_irradiance_map
            * dni_to_hadlow_adjustment_factor.reset_index(drop=True)
        )

    # Compute the amount of diffuse light reaching the ground.
    with time_execution("Diffuse on-the-ground calculation"):
        # Compute the diffuse irradiance on the ground.
        clearsky_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
            {
                ground_index: (
                    clearsky_total_diffuse_surface_irradiance.reset_index(drop=True)
                    * polytunnel.transmissivity
                    * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
                        drop=True
                    )
                    * ground_to_surface_projection_frame.iloc[ground_index]
                ).sum(axis=1)
                for ground_index, _ in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Ground diffuse-irradiance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            }
        )

        # Compute the diffuse irradiance on the ground given diffuse and direct
        # assumptions for the Hadlow data
        diffuse_day_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
            {
                ground_index: (
                    diffuse_day_total_diffuse_surface_irradiance.reset_index(drop=True)
                    * polytunnel.transmissivity
                    * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
                        drop=True
                    )
                    * ground_to_surface_projection_frame.iloc[ground_index]
                ).sum(axis=1)
                for ground_index, _ in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Diffuse day ground-irradiance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            }
        )

        direct_day_ground_diffuse_irradiance_map: pd.DataFrame = pd.DataFrame(
            {
                ground_index: (
                    direct_day_total_diffuse_surface_irradiance.reset_index(drop=True)
                    * polytunnel.transmissivity
                    * polytunnel_surface_pv_uncovered_fraction_mask.reset_index(
                        drop=True
                    )
                    * ground_to_surface_projection_frame.iloc[ground_index]
                ).sum(axis=1)
                for ground_index, _ in tqdm(
                    enumerate(polytunnel.ground_mesh),
                    desc="Direct day ground-irradiance calculation",
                    leave=False,
                    total=len(polytunnel.ground_mesh),
                )
            }
        )

        # TODO: Implement diffuse light from the ends of the polytunnel.

    # Readjust the direct day to compute a direct-beam irradiance.
    direct_day_ground_direct_beam_irradiance: pd.DataFrame = pd.DataFrame(
        direct_day_ground_direct_irradiance.values
        / pd.DataFrame(
            [cos(position.theta_spherical) for position in solar_positions]
        ).values,
        index=direct_day_ground_direct_irradiance.index,
        columns=direct_day_ground_direct_irradiance.columns,
    )

    # Compute the total on-the-ground irradiance map
    with time_execution("Global on-the-ground calculation"):
        # Compute a generalised on-the-ground map for clearsky conditions.
        clearsky_total_ground_irradiance_map: pd.DataFrame = (
            clearsky_ground_direct_irradiance_map.reset_index(drop=True)
            + clearsky_ground_diffuse_irradiance_map.reset_index(drop=True)
        )

        # Compute a map where all irradiance on the surface is taken to be direct
        # solar irradiance but is scaled by the Hadlow numbers.
        direct_day_total_ground_irradiance_map: pd.DataFrame = (
            direct_day_ground_diffuse_irradiance_map.reset_index(drop=True)
            + direct_day_ground_direct_irradiance.reset_index(drop=True)
        )

        direct_day_total_ground_with_beam_irradiance_map: pd.DataFrame = (
            direct_day_ground_diffuse_irradiance_map.reset_index(drop=True)
            + direct_day_ground_direct_beam_irradiance.reset_index(drop=True)
        )

        diffuse_day_total_ground_irradiance_map: pd.DataFrame = (
            diffuse_day_ground_diffuse_irradiance_map.reset_index(drop=True)
        )

    # Reset missing indices
    clearsky_total_ground_irradiance_map.index = hadlow_dni_slice.index
    clearsky_ground_direct_irradiance_map.index = hadlow_dni_slice.index
    clearsky_ground_diffuse_irradiance_map.index = hadlow_dni_slice.index

    direct_day_ground_diffuse_irradiance_map.index = hadlow_dni_slice.index
    direct_day_ground_direct_irradiance.index = hadlow_dni_slice.index
    direct_day_ground_direct_beam_irradiance.index = hadlow_dni_slice.index
    direct_day_total_ground_irradiance_map.index = hadlow_dni_slice.index
    direct_day_total_ground_with_beam_irradiance_map.index = hadlow_dni_slice.index

    diffuse_day_ground_diffuse_irradiance_map.index = hadlow_dni_slice.index
    diffuse_day_total_ground_irradiance_map.index = hadlow_dni_slice.index

    # Parse the validation data if provided to compare against.
    if parsed_args.validation_filename is not None:
        with time_execution("Generating validation plots"):
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
                0.1 * validation_data[ValidationColumns.DIFFUSE_PAR.value]
            )
            validation_data[ValidationColumns.DIRECT_ERROR.value] = (
                0.1 * validation_data[ValidationColumns.DIRECT_PAR.value]
            )
            validation_data[ValidationColumns.TOTAL_ERROR.value] = (
                0.1 * validation_data[ValidationColumns.TOTAL_PAR.value]
            )

            dir_day_gnd_tot_val: pd.DataFrame = pd.merge(
                direct_day_total_ground_with_beam_irradiance_map,
                validation_data,
                left_index=True,
                right_index=True,
            )
            dir_day_gnd_dir_val: pd.DataFrame = pd.merge(
                direct_day_ground_direct_beam_irradiance,
                validation_data,
                left_index=True,
                right_index=True,
            )
            dir_day_gnd_dif_val: pd.DataFrame = pd.merge(
                direct_day_ground_diffuse_irradiance_map,
                validation_data,
                left_index=True,
                right_index=True,
            )

            dif_day_gnd_tot_val: pd.DataFrame = pd.merge(
                diffuse_day_total_ground_irradiance_map,
                validation_data,
                left_index=True,
                right_index=True,
            )

            try:
                sns.set_palette(
                    # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
                    ["#423252", "#4A688B", "#779FB1", "#36C7B8", "#FBC412"],
                )
            except UnboundLocalError:
                import seaborn as sns
                import matplotlib.pyplot as plt

                sns.set_palette(
                    # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
                    ["#423252", "#4A688B", "#779FB1", "#36C7B8", "#FBC412"],
                )

            #######################
            # Plotting code No. 1 #
            #######################

            # import matplotlib.pyplot as plt
            # import matplotlib.animation as animation
            # import seaborn as sns
            # import numpy as np

            # fig, ax = plt.subplots(figsize=(180*MM, 120*MM))

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
            # ani.save("diffuse_day_total_ground_irradiance_map_2.gif", writer="pillow", fps=15)
            # plt.show()

            #######################
            # Plotting code No. 3 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
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
                y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                color="C0",
                label="Total PAR",
                marker="h",
                s=40,
            )
            plt.plot(
                dir_day_gnd_tot_val.index,
                dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                color="C0",
            )
            plt.errorbar(
                dir_day_gnd_dir_val.index,
                dir_day_gnd_dir_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value] / 2.1,
                ls="none",
                color="C0",
            )
            plt.xlabel("Date and time")
            plt.ylabel("Irradiance / W/m$^2$")

            axis_right = (axis_left := plt.gca()).twinx()
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
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)

            plt.savefig(
                f"validation_{parsed_args.validation_index}_total_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_"
                f"{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            #######################
            # Plotting code No. 4 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
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
                y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                color="C1",
                label="Direct PAR",
                marker="h",
                s=40,
            )
            plt.plot(
                dir_day_gnd_dir_val.index,
                dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                color="C1",
            )
            plt.errorbar(
                dir_day_gnd_dir_val.index,
                dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.DIRECT_ERROR.value] / 2.1,
                ls="none",
                color="C1",
            )

            plt.xlabel("Date and time")
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

            plt.legend(
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)
            plt.savefig(
                f"validation_{parsed_args.validation_index}_direct_diff_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            #######################
            # Plotting code No. 5 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
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
                y=dif_day_gnd_tot_val[parsed_args.validation_index],
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
                y=dir_day_gnd_dif_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                color="C1",
                label="Diffuse PAR",
                marker="h",
                s=40,
            )
            plt.plot(
                dir_day_gnd_dif_val.index,
                dir_day_gnd_dif_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                color="C1",
            )
            plt.errorbar(
                dir_day_gnd_dir_val.index,
                dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_ERROR.value] / 2.1,
                ls="none",
                color="C1",
            )
            plt.xlabel("Date and time")
            plt.ylabel("Irradiance / W/m$^2$")

            plt.legend()

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

            plt.legend(
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)
            plt.savefig(
                f"validation_{parsed_args.validation_index}_diffuse_diff_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_"
                f"{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            #######################
            # Plotting code No. 6 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
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
                y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                color="C0",
                label="Total PAR",
                marker="h",
                s=60,
                zorder=1,
            )
            plt.plot(
                range(len(dir_day_gnd_tot_val)),
                dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                color="C0",
                zorder=1,
            )
            plt.errorbar(
                x=range(len(dir_day_gnd_tot_val)),
                y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value] / 2.1,
                ls="none",
                color="C0",
                zorder=1,
            )
            plt.xlabel("Date and time")
            plt.ylabel("Irradiance / W/m$^2$")

            axis_right = (axis_left := plt.gca()).twinx()
            axis_left.tick_params(axis="both", which="major", labelsize=7)
            axis_right.tick_params(axis="both", which="major", labelsize=7)
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
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)

            plt.savefig(
                "validation_total_map_boxplot_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_"
                f"{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            #######################
            # Plotting code No. 7 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
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
                y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                color="C1",
                label="Diffuse PAR",
                marker="h",
                s=60,
                zorder=1,
            )
            plt.plot(
                range(len(dir_day_gnd_dir_val)),
                dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                color="C1",
                zorder=1,
            )
            plt.errorbar(
                x=range(len(dir_day_gnd_dir_val)),
                y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_ERROR.value] / 2.1,
                ls="none",
                color="C1",
                zorder=1,
            )
            plt.xlabel("Date and time")
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

            plt.legend(
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)

            plt.savefig(
                "validation_diffuse_map_boxplot_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_"
                f"{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            #######################
            # Plotting code No. 8 #
            #######################

            plt.figure(figsize=(180 * MM, 120 * MM))
            sns.scatterplot(
                x=range(len(dir_day_gnd_dir_val)),
                y=dir_day_gnd_dir_val.reset_index(drop=True).transpose()[:-13].mean(axis=0),
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
                dir_day_gnd_dir_val.reset_index(drop=True).transpose()[:-13].mean(axis=0),
                color="C4",
                zorder=0,
            )
            sns.scatterplot(
                x=range(len(dir_day_gnd_dir_val)),
                y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                color="C1",
                label="Direct PAR",
                marker="h",
                s=60,
                zorder=1,
            )
            plt.plot(
                range(len(dir_day_gnd_dir_val)),
                dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                color="C1",
                zorder=1,
            )
            plt.errorbar(
                x=range(len(dir_day_gnd_dir_val)),
                y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] / 2.1,
                yerr=dir_day_gnd_dir_val[ValidationColumns.DIRECT_ERROR.value] / 2.1,
                ls="none",
                color="C1",
                zorder=1,
            )
            plt.xlabel("Date and time")
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

            plt.legend(
                left_handles + right_handles, left_labels + right_labels, loc="upper right"
            )
            axis_right.set_ylim(-0.05, 1.05)
            axis_left.set_ylim(-25, 825)

            plt.savefig(
                "validation_direct_map_boxplot_"
                f"{diffusivity}_{polytunnel.name}_"
                f"{parsed_args.start_time.replace(':','_')}_"
                f"{parsed_args.end_time.replace(':','_')}.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            # plt.show()

    # import pdb

    # pdb.set_trace()

    return

    sns.scatterplot(
        x=dif_day_gnd_tot_val.index,
        y=dif_day_gnd_tot_val[parsed_args.validation_index],
        marker="h",
    )
    sns.scatterplot(
        x=dif_day_gnd_tot_val.index,
        y=dif_day_gnd_tot_val[ValidationColumns.DIFFUSE_PAR.value],
        marker="h",
    )

    # combined_frame = pd.merge()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns
    import numpy as np

    fig, ax = plt.subplots()

    # Create initial heatmap with dummy data
    initial_data = np.reshape(
        direct_day_total_ground_irradiance_map.iloc[0],
        (
            _dim_x := polytunnel.meshgrid_resolution,
            _dim_y := polytunnel.length_wise_meshgrid_resolution,
        ),
    )
    vmin = 0
    vmax = max(direct_day_total_ground_irradiance_map.max(axis=0))
    heatmap = sns.heatmap(
        initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax
    )

    _ten_minutes: int = int(
        _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
    )

    def update(time_index: int):
        ax.clear()  # clear previous heatmap
        data = np.reshape(
            direct_day_total_ground_irradiance_map.iloc[time_index], (_dim_x, _dim_y)
        )
        sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
        ax.set_title(
            f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
        )

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(direct_day_total_ground_irradiance_map),
        interval=300,
        repeat=False,
    )
    ani.save("16_may_24_control_diffuse_ground_t_0.5.gif", writer="pillow", fps=15)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns
    import numpy as np

    fig, ax = plt.subplots()

    # Create initial heatmap with dummy data
    initial_data = np.reshape(
        direct_day_total_ground_irradiance_map.iloc[0],
        (
            _dim_x := polytunnel.meshgrid_resolution,
            _dim_y := polytunnel.length_wise_meshgrid_resolution,
        ),
    )
    vmin = 0
    vmax = max(direct_day_total_ground_irradiance_map.max(axis=0))
    heatmap = sns.heatmap(
        initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax
    )

    _ten_minutes: int = int(
        _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
    )

    def update(time_index: int):
        ax.clear()  # clear previous heatmap
        data = np.reshape(
            direct_day_total_ground_irradiance_map.iloc[time_index], (_dim_x, _dim_y)
        )
        sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
        ax.set_title(
            f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
            f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
        )

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(direct_day_total_ground_irradiance_map),
        interval=300,
        repeat=False,
    )
    ani.save("direct_day_total_ground_irradiance_map.gif", writer="pillow", fps=15)
    plt.show()

    fig, ax = plt.subplots()

    # Create initial heatmap with dummy data
    initial_data = np.reshape(
        diffuse_day_total_ground_irradiance_map.iloc[0],
        (
            _dim_x := polytunnel.meshgrid_resolution,
            _dim_y := polytunnel.length_wise_meshgrid_resolution,
        ),
    )
    vmin = 0
    vmax = max(diffuse_day_total_ground_irradiance_map.max(axis=0))
    heatmap = sns.heatmap(
        initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax
    )

    _ten_minutes: int = int(
        _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
    )

    def update(time_index: int):
        ax.clear()  # clear previous heatmap
        data = np.reshape(
            diffuse_day_total_ground_irradiance_map.iloc[time_index], (_dim_x, _dim_y)
        )
        sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
        ax.set_title(
            f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
        )

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(diffuse_day_total_ground_irradiance_map),
        interval=300,
        repeat=False,
    )
    ani.save("diffuse_day_total_ground_irradiance_map.gif", writer="pillow", fps=15)
    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.heatmap(surface_shaded_map)
    plt.savefig(
        "surface_shaded_unmodulatead.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()

    sns.heatmap(direct_surface_irradiance, cmap="viridis")
    plt.savefig(
        "direct_surface_unmodulatead.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()

    sns.heatmap(
        np.reshape(
            [meshpoint.covered_fraction for meshpoint in polytunnel.surface_mesh],
            (parsed_args.meshgrid_resolution, parsed_args.meshgrid_resolution),
        )
    )
    plt.savefig(
        "polytunnel_positions.pdf", format="pdf", bbox_inches="tight", pad_inches=0
    )
    plt.show()

    sun_not_shaded = pd.DataFrame(
        {
            meshpoint_index: [
                calculate_adjacent_polytunnel_shading(
                    meshpoint, polytunnel, solar_position
                )
                for solar_position in solar_positions
            ]
            for meshpoint_index, meshpoint in enumerate(polytunnel.surface_mesh)
        }
    )

    plt.figure()
    plt.title("Sun not shaded")
    sns.heatmap(sun_not_shaded)
    plt.show()

    for time_index, irradiance in direct_surface_irradiance.iterrows():
        if irradiance.sum(axis=0) == 0:
            continue
        plt.figure()
        sns.heatmap(
            np.reshape(irradiance, (10, 10)),
            cmap="viridis",
            vmin=0,
            vmax=max(direct_surface_irradiance.max(axis=0)),
        )
        plt.title(
            f"Time index: {time_index}. Date: {time_index // (6 * 24)}; Time: {time_index // 6}:{time_index % 6}0"
        )
        plt.show()

    for time_index, sun_not_shaded_row in sun_not_shaded.iterrows():
        plt.figure()
        sns.heatmap(
            np.reshape(sun_not_shaded_row, (10, 10)),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.title(
            f"Time index: {time_index}. Date: {time_index // (6 * 24)}; Time: {time_index // 6}:{time_index % 6}0"
        )
        plt.show()

    plt.figure()
    plt.title("Sun above horizon")
    sns.heatmap(
        pd.DataFrame(
            {
                meshpoint_index: [
                    (solar_position.z > 0) for solar_position in solar_positions
                ]
                for meshpoint_index, meshpoint in enumerate(polytunnel.surface_mesh)
            }
        )
    )
    plt.show()

    plt.figure()
    plt.title("Sun shining on polytunnel meshpoint")
    sns.heatmap(
        sun_shining := pd.DataFrame(
            {
                meshpoint_index: [
                    ((meshpoint._normal_vector * solar_position) > 0)
                    for solar_position in solar_positions
                ]
                for meshpoint_index, meshpoint in enumerate(polytunnel.surface_mesh)
            }
        )
    )
    plt.show()

    plt.figure()
    plt.scatter(range(144), [position.phi for position in solar_positions])
    plt.axhline(
        polytunnel.surface_mesh[0].intercept_plane._first_vector.phi, color="blue"
    )
    plt.axhline(
        polytunnel.surface_mesh[0].intercept_plane._second_vector.phi, color="red"
    )
    plt.plot(range(144), sun_shining[0], "--")
    plt.plot(range(144), sun_not_shaded[0], "-.")
    plt.show()

    plt.figure()
    plt.scatter(range(144), [position.phi for position in solar_positions])
    plt.axhline(
        polytunnel.surface_mesh[-1].intercept_plane._first_vector.phi, color="blue"
    )
    plt.axhline(
        polytunnel.surface_mesh[-1].intercept_plane._second_vector.phi, color="red"
    )
    plt.plot(range(144), sun_shining[99], "--")
    plt.plot(range(144), sun_not_shaded[99], "-.")
    plt.show()

    d = 2 * semi_major_axis
    # sun transits#
    sun = Sun(
        start_time=start_time_str,
        end_time=end_time_str,
        latitude=latitude,
        longitude=longitude,
        resolution_minutes=res_minutes,
    )
    altitude_array, azimuth_array = sun.generate_sun_positions()
    time_array = sun.get_times()

    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)
    sun_surface_grid, sun_incident = sun.sunvec_tilts_grid(
        sun_vecs, normals_unit_surface
    )

    # Used for Diffuse Component
    clear_sky = sun.get_clearsky_data_new(
        start_time_str, end_time_str, latitude, longitude, res_minutes
    )

    sun_positions_calculation_time = time.time()
    print(
        f"Sun positions calculation time: { sun_positions_calculation_time - end_geometry_time } seconds"
    )

    # shading#
    tracer = Tracing(
        tunnel, sun_vecs, surface_grid, tilts_unit, semi_major_axis, semi_minor_axis
    )
    gradient_grid, angle_grid, surface_gradient_grid, surface_angle_grid = (
        tracer.find_tangent_gradient()
    )

    shading_calculation_time = time.time()
    print(
        f"Shading calculation time: { shading_calculation_time - sun_positions_calculation_time } seconds"
    )

    irradiance = TunnelIrradiance(tunnel, semi_major_axis, semi_minor_axis, length)
    shaded_exposure_map = irradiance.shading_exposure_map(
        angle_grid, surface_angle_grid, sun_vecs
    )

    # zero tilt, PAR region: 400-700
    # optical_wavelengths in nm
    # optical intensities in W/m^2/nm
    # spectra_frames units in W/m^2
    optical_wavelengths, optical_intensities, spectra_frames = sun.get_spectra(400, 700)

    wavelengths_intensities_spectra_frames_calculation_time = time.time()
    print(
        f"Wavelengths, Intensities and Spectral Frames calculation time: { wavelengths_intensities_spectra_frames_calculation_time - shading_calculation_time  } seconds"
    )

    # TMM#
    print("Performing TMM calculation")

    if multistack == 1:
        # Active Layer: Bilayer Structure
        material_list = material_list
        material_thickness = np.array(["inf"] + material_thick + ["inf"])
    elif multistack > 1:
        # Active layer: Multistack Structure, Overlaping multistack times Donor And Acceptor Layers
        # The thickness of the Active Layer will be donor thickness + acceptor thickness defined in
        # parameters csv file
        material_list = (
            material_list[:2]
            + [material_list[2], material_list[3]] * multistack
            + material_list[4:]
        )
        material_thickness = np.array(
            ["inf"]
            + material_thick[:2]
            + [material_thick[2] / multistack, material_thick[3] / multistack]
            * multistack
            + material_thick[4:]
            + ["inf"]
        )

    tmm_calculation_time = time.time()
    print(
        f"TMM calculation time: { tmm_calculation_time - wavelengths_intensities_spectra_frames_calculation_time  } seconds"
    )

    # Performing TMM, complex array contains N,K values for every material in material list
    complex_array = tracer.n_list_wavelength(material_list, optical_wavelengths)

    # Transmitted ligth at every time
    t_grid_frames = irradiance.t_grid(
        sun_incident, optical_wavelengths, complex_array, material_thickness
    )

    # Solar Spectra#
    print("Performing Solar Spectra calculation")
    # Obtain the irradiance rays affected by transmittance and shading
    # solar_cell_spectra has units of W/m^2/nm
    # solar_cell_spectra_shaded has units of W/m^2/nm
    solar_cell_spectra = irradiance.solar_cells_irradiance_rays(
        optical_intensities, t_grid_frames, solar_cells
    )
    solar_cell_spectra_shaded = irradiance.shaded_irradiance_spectra(
        solar_cell_spectra, shaded_exposure_map
    )

    solar_spectra_calculation_time = time.time()
    print(
        f"Solar spectra calculation time: { solar_spectra_calculation_time - tmm_calculation_time } seconds"
    )

    # Irradiance#
    print("Performing Irradiance calculation")
    # Calculate irradiance after roof of polytunnel by considering
    # the roof cover by polyethene, in any other case, find N,K for
    # material of the polytunnel
    # Transmmited_frames contains irradiance in the interior of polytunnel
    # transmitted_frames, absorbed_frames have units of W/m^2/nm
    transmitted_frames, absorbed_frames = irradiance.transmitted_absorbed_spectra(
        optical_wavelengths, solar_cell_spectra_shaded, sun_surface_grid
    )

    # scale data with spectral sensitivity of BF5 sensor:
    transmitted_frames = irradiance.scaled_spectra_sensor_bf5(
        optical_wavelengths, transmitted_frames
    )
    absorbed_frames = irradiance.scaled_spectra_sensor_bf5(
        optical_wavelengths, absorbed_frames
    )

    # transmitted_int, absorbed_int have units of W/m^2
    transmitted_int = irradiance.int_spectra(optical_wavelengths, transmitted_frames)
    absorbed_int = irradiance.int_spectra(optical_wavelengths, absorbed_frames)

    # Irradiance at interior of polytunel on every grid of polytunnel
    # Absorbed irradiance at interior of polytunnel on every grid of polytunnel
    _ = viz.every_grid_plot(
        transmitted_frames,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/transmited_ligth_{parameters_sim}.png",
    )
    _ = viz.every_grid_plot(
        absorbed_frames,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/absorbed_ligth_{parameters_sim}.png",
    )

    transmitted_spectra_calculation_time = time.time()
    print(
        f"Solar spectra calculation time: { transmitted_spectra_calculation_time - solar_spectra_calculation_time } seconds"
    )

    # Direct Irradiance (ground)#
    print("Performing Solar Spectra calculation Direct")
    # direct component of irradiance (ground) with units of W/m^2/nm
    direct_ground_irradiance_frames = irradiance.direct_irradiance_spectra_ground(
        ground_grid,
        normals_unit_ground,
        surface_grid,
        distance_grid,
        sun_vecs,
        transmitted_frames,
    )
    # direct component of irradiance integrated over wavelengths, with unitsof W/m^2
    direct_ground_int = irradiance.int_spectra(
        optical_wavelengths, direct_ground_irradiance_frames
    )

    direct_irradiance_calculation_time = time.time()
    print(
        f"Direct ground irradiance calculation time: { direct_irradiance_calculation_time - transmitted_spectra_calculation_time  } seconds"
    )

    # Diffuse Irradiance (ground)#
    print("Performing Solar Spectra calculation Diffuse")
    # diffuse component of irradiance (ground) with units of W/m^2/nm
    # obtained from absorbed frames on polyethene,
    diffuse_ground_irradiance_frames = irradiance.diffuse_irradiance_spectra_ground(
        distance_grid,
        separation_unit_vector_grid,
        normals_unit_ground,
        normals_unit_surface,
        areas_surface,
        absorbed_frames,
    )
    # diffuse component of irradiance integrated over wavelengthswith units of W/m^2
    diffuse_ground_int = irradiance.int_spectra(
        optical_wavelengths, diffuse_ground_irradiance_frames
    )

    diffuse_irradiance_calculation_time = time.time()
    print(
        f"Diffuse ground irradiance calculation time: { diffuse_irradiance_calculation_time - transmitted_spectra_calculation_time  } seconds"
    )

    # Global Irradiance (ground)#
    print("Performing Solar Spectra calculation GLOBAL GROUND")
    # global component of irradiance (ground) with units of W/m^2/nm
    global_ground_irradiance_frames = irradiance.global_irradiance_spectra_ground(
        diffuse_ground_irradiance_frames, direct_ground_irradiance_frames
    )
    # global component of irradiance integrated over wavelengthswith units of W/m^2
    global_ground_int = irradiance.int_spectra(
        optical_wavelengths, global_ground_irradiance_frames
    )

    global_ground_irradiance_calculation_time = time.time()
    print(
        f"Global ground irradiance calculation time: { global_ground_irradiance_calculation_time  - diffuse_irradiance_calculation_time } seconds"
    )

    # Photon spectra PAR Direct#
    print("Performing PAR calculation DIRECT")
    direct_photon_spectra_frames = irradiance.power_to_photon_spectra(
        optical_wavelengths, direct_ground_irradiance_frames
    )
    direct_photon_par_spectra = irradiance.par_spectra(
        optical_wavelengths, direct_photon_spectra_frames
    )
    direct_photon_par = irradiance.int_spectra(
        optical_wavelengths, direct_photon_par_spectra
    )
    par_direct_calculation_time = time.time()

    print(
        f"PAR Direct calculation time: { par_direct_calculation_time - global_ground_irradiance_calculation_time } seconds"
    )

    # avg_value, std_value, max_value = compute_region_means(direct_photon_par)

    # avg_value_irr, std_value_irr, max_value_irr = compute_region_means(direct_ground_int)

    # match_data = pd.DataFrame( { 'Time'           : time_array,
    #                              'amax_par_direct' : max_value[3],
    #                              'astd_par_direct'  : std_value[3],
    #                              'aavg_par_direct'  : avg_value[3],
    #                              'rmax_par_direct' : max_value[2],
    #                              'rstd_par_direct'  : std_value[2],
    #                              'ravg_par_direct'  : avg_value[2],
    #                              'cmax_par_direct' : max_value[1],
    #                              'cstd_par_direct'  : std_value[1],
    #                              'cavg_par_direct'  : avg_value[1],
    #                              'lmax_par_direct' : max_value[0],
    #                              'lstd_par_direct'  : std_value[0],
    #                              'lavg_par_direct'  : avg_value[0],
    #                              'amax_irradiance_ground_direct' : max_value_irr[3],
    #                              'astd_irradiance_ground_direct'  : std_value_irr[3],
    #                              'aavg_irradiance_ground_direct'  : avg_value_irr[3],
    #                              'rmax_irradiance_ground_direct' : max_value_irr[2],
    #                              'rstd_irradiance_ground_direct'  : std_value_irr[2],
    #                              'ravg_irradiance_ground_direct'  : avg_value_irr[2],
    #                              'cmax_irradiance_ground_direct' : max_value_irr[1],
    #                              'cstd_irradiance_ground_direct'  : std_value_irr[1],
    #                              'cavg_irradiance_ground_direct'  : avg_value_irr[1],
    #                              'lmax_irradiance_ground_direct' : max_value_irr[0],
    #                              'lstd_irradiance_ground_direct'  : std_value_irr[0],
    #                              'lavg_irradiance_ground_direct'  : avg_value_irr[0],
    #                            } )

    # match_data.to_csv(f'match_data_direct_photonpar_{parameters_sim}.csv', index = False )

    # Photon spectra PAR Diffuse#
    print("Performing PAR calculation DIFFUSE")
    diffuse_photon_spectra_frames = irradiance.power_to_photon_spectra(
        optical_wavelengths, diffuse_ground_irradiance_frames
    )
    diffuse_photon_par_spectra = irradiance.par_spectra(
        optical_wavelengths, diffuse_photon_spectra_frames
    )
    diffuse_photon_par = irradiance.int_spectra(
        optical_wavelengths, diffuse_photon_par_spectra
    )

    par_diffuse_calculation_time = time.time()
    print(
        f"PAR Diffuse calculation time: { par_diffuse_calculation_time -  par_direct_calculation_time } seconds"
    )

    # Photon spectra PAR Global (ground)#
    print("Performing PAR calculation GLOBAL")
    global_photon_spectra_frames = irradiance.power_to_photon_spectra(
        optical_wavelengths, global_ground_irradiance_frames
    )
    global_photon_par_spectra = irradiance.par_spectra(
        optical_wavelengths, global_photon_spectra_frames
    )
    global_photon_par = irradiance.int_spectra(
        optical_wavelengths, global_photon_par_spectra
    )

    par_global_calculation_time = time.time()
    print(
        f"PAR Global calculation time: { par_global_calculation_time - par_diffuse_calculation_time  } seconds"
    )

    print("Photon PAR:")
    direct_data = viz.every_grid_plot(
        direct_photon_spectra_frames,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/direct_photon_spectra_{parameters_sim}.png",
    )
    diffuse_data = viz.every_grid_plot(
        diffuse_photon_spectra_frames,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/diffuse_photon_spectra_{parameters_sim}.png",
    )
    global_data = viz.every_grid_plot(
        global_photon_spectra_frames,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/global_photon_spectra_{parameters_sim}.png",
    )

    par_direct_data = viz.every_grid_plot(
        direct_photon_par_spectra,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/par_direct_photon_spectra_{parameters_sim}.png",
    )
    par_diffuse_data = viz.every_grid_plot(
        diffuse_photon_par_spectra,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/par_diffuse_photon_spectra_{parameters_sim}.png",
    )
    par_global_data = viz.every_grid_plot(
        global_photon_par_spectra,
        time_array,
        optical_wavelengths,
        res_minutes,
        f"figures/Date_{date_simulation}/par_global_photon_spectra_{parameters_sim}.png",
    )

    match_data = pd.DataFrame(
        {
            "Time": time_array,
            "all_direct": direct_data[0],
            "cen_direct": direct_data[1],
            "lef_direct": direct_data[2],
            "rig_direct": direct_data[3],
            "all_diffuse": diffuse_data[0],
            "cen_diffuse": diffuse_data[1],
            "lef_diffuse": diffuse_data[2],
            "rig_diffuse": diffuse_data[3],
            "all_global": global_data[0],
            "cen_global": global_data[1],
            "lef_global": global_data[2],
            "rig_global": global_data[3],
            "all_par_direct": par_direct_data[0],
            "cen_par_direct": par_direct_data[1],
            "lef_par_direct": par_direct_data[2],
            "rig_par_direct": par_direct_data[3],
            "all_par_diffuse": par_diffuse_data[0],
            "cen_par_diffuse": par_diffuse_data[1],
            "lef_par_diffuse": par_diffuse_data[2],
            "rig_par_diffuse": par_diffuse_data[3],
            "all_par_global": par_global_data[0],
            "cen_par_global": par_global_data[1],
            "lef_par_global": par_global_data[2],
            "rig_par_global": par_global_data[3],
        }
    )

    match_data.to_csv(
        f"Matched_Dates/match_data_direct_photonpar_{parameters_sim}.csv", index=False
    )

    # viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, global_photon_par_spectra, f"figures/Date_{date_simulation}/TESTGLOBALPARNOFRAMES-surface-animation_{parameters_sim}.mp4")
    # viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, global_photon_spectra_frames, f"figures/Date_{date_simulation}/TESTGLOBALPAR-surface-animation_{parameters_sim}.mp4")

    diffuse_irradiance_frames = irradiance.diffuse_irradiance_ground(
        distance_grid,
        separation_unit_vector_grid,
        normals_unit_ground,
        normals_unit_surface,
        areas_surface,
        diffuse_ground_irradiance_frames,
        transmissivity,
    )
    direct_irradiance_frames = irradiance.direct_irradiance_ground(
        normals_unit_ground, sun_vecs, direct_ground_irradiance_frames, transmissivity
    )

    power_total_ground_diffuse = irradiance.power(
        areas_ground, diffuse_irradiance_frames
    )
    # power_total_surface = irradiance.power(areas_surface, irradiance_frames)
    # power_total_surface = irradiance.power(areas_surface, direct_irradiance_frames)
    power_total_ground_direct = irradiance.power(areas_ground, direct_irradiance_frames)
    power_total_out = np.array(power_total_ground_diffuse) + np.array(
        power_total_ground_direct
    )

    viz.plot_sun(
        time_array,
        altitude_array,
        azimuth_array,
        spectra_frames,
        f"figures/Date_{date_simulation}/sun_{parameters_sim}.png",
    )

    # viz.plot_surface(surface_grid_x, surface_grid_y, surface_grid_z, normals_unit_surface, ground_grid_x, ground_grid_y, ground_grid_z, normals_unit_ground, sun_vec=sun_vecs[72])

    # viz.plot_irradiance(surface_grid_x, surface_grid_y, irradiance_frames[60])

    viz.animate_irradiance(
        time_array,
        surface_grid_x,
        surface_grid_y,
        transmitted_int,
        "Irradiance [W/m^2]",
        f"figures/Date_{date_simulation}/direct-irradiance-surface-animation_{parameters_sim}.mp4",
    )

    viz.animate_irradiance(
        time_array,
        ground_grid_x,
        ground_grid_y,
        direct_ground_int,
        "Irradiance [W/m^2]",
        f"figures/Date_{date_simulation}/direct-irradiance-ground-animation_{parameters_sim}.mp4",
    )

    # viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, diffuse_ground_int, f"figures/Date_{date_simulation}/diffuse-irradiance-ground-animation_{parameters_sim}.mp4")

    # viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, global_ground_int, f"figures/Date_{date_simulation}/global-irradiance-ground-animation_{parameters_sim}.mp4")

    # viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, photon_par, "figures/photon-par-animation.mp4")
    viz.animate_irradiance(
        time_array,
        ground_grid_x,
        ground_grid_y,
        direct_photon_par,
        "Irradiance [W/m^2]",
        f"figures/Date_{date_simulation}/direct_photon_par-animation_{parameters_sim}.mp4",
    )

    total_time = time.time()
    print(f"Total time { total_time - start_simulation_time }")

    # viz.plot_power(time_array, power_total_ground_diffuse, power_total_ground_direct, power_total_out, f'figures/Date_{date_simulation}/power-received_{parameters_sim}.png')


#######################
# Plotting code No. 1 #
#######################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# fig, ax = plt.subplots()

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     direct_day_diffuse_surface_irradiance.iloc[0],
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = max(direct_day_diffuse_surface_irradiance.max(axis=0))
# heatmap = sns.heatmap(
#     initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )


# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         direct_day_diffuse_surface_irradiance.iloc[time_index], (_dim_x, _dim_y)
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )


# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(direct_day_diffuse_surface_irradiance),
#     interval=300,
#     repeat=False,
# )
# ani.save("direct_day_diffuse_surface_irradiance.gif", writer="pillow", fps=15)
# plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
