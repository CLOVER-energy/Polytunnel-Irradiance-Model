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
import os
import sys
import time

from contextlib import contextmanager
from typing import Any, Callable, Generator

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.polytunnel_irradiance_model.__utils__ import *
from src.polytunnel_irradiance_model.functions import *
from src.polytunnel_irradiance_model.geometry import Polytunnel
from src.polytunnel_irradiance_model.sun import Sun
from src.polytunnel_irradiance_model.irradiance import TunnelIrradiance
from src.polytunnel_irradiance_model.tracing import Tracing
import src.polytunnel_irradiance_model.visualisation as viz


import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r".*vectorized_tmm_dispersive_multistack.*"
)

__all__ = ("compute_surface_grid", "main")


# DONE:
#   Snippet to print when code is succesffully executed.
DONE: str = "[  DONE  ]"

# FAILED:
#   Snippet to print when code fails to execute.
FAILED: str = "[ FAILED ]"

# MODULES:
#   Keyword for parsing PV-module information.
MODULES: str = "modules"


def code_print(string_to_print: str) -> None:
    """
    Print a line with dots.

    :param: string_to_print:
        The string to print.

    """

    print(string_to_print + "." * (65 - len(string_to_print)) + " ", end="")


def compute_surface_grid():
    """Returns the surface grid"""


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
        "--modelling-temporal-resolution",
        "-rtm",
        type=float,
        default=10,
        help="The temporal resolution, in minutes, to use when simulating throughout "
        "the day.",
    )

    # Polytunnel arguments
    polytunnel_arguments = parser.add_argument_group(
        "polytunnel arguments",
        description="Arguments used to specify the technical details of the polytunnel.",
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
        type=int,
        default=10,
        help="The resolution of the mesh grid in terms of the number of points along "
        "each dimension of the polytunnel to use; default of 10.",
    )

    return parser.parse_args(args)


@contextmanager
def time_execution() -> Generator[Callable[[], float], Any, None]:
    """
    Times a period of code execution.

    :yields:
        The elapsed time taken for the code to execute.

    """

    start_time: float = time.perf_counter()
    end_time: float | None = None

    try:
        yield lambda: (
            end_time - start_time
            if end_time is not None
            else time.perf_counter() - start_time
        )
    finally:
        end_time = time.perf_counter()


def main(args: list[Any]) -> None:
    """
    Main function for operating the Polytunnel-Irradiance-Model.

    :param: args:
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = parse_args(args)

    global MESHGRID_RESOLUTION
    MESHGRID_RESOLUTION = parsed_args.meshgrid_resolution

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
    simulation_datetime = datetime.datetime.strptime(
        parsed_args.start_time, "%Y-%m-%dT%H:%M:%SZ"
    )

    if not os.path.isdir(
        output_figures_dir := os.path.join(
            "output_files", "figures", simulation_datetime.strftime("%Y_%m_%d")
        )
    ):
        os.makedirs(output_figures_dir)

    code_print("Geometry calculation")
    try:
        with time_execution() as geometry_timer:
            polytunnel = Polytunnel.from_data(polytunnel_data, pv_module_inputs)

            import pdb

            pdb.set_trace()

            tunnel = ElipticalPolytunnel(
                length=length,
                tilt=tilt,
                azimuthal_orientation=azimuthal_orientation,
                theta_margin=0,
                cell_thickness=cell_thickness,
                cell_spacing=cell_spacing,
                semi_major_axis=parsed_args.semi_major_axis,
                semi_minor_axis=parsed_args.semi_minor_axis,
                meshgrid_resolution=meshgrid_resolution,
                initial_cell_spacing=initial_cell_spacing,
            )
            tunnel_l = ElipticalPolytunnel(
                semi_major_axis=semi_major_axis,
                length=length,
                tilt=tilt,
                azimuthal_orientation=azimuthal_orientation,
                x_shift=3.0,
                meshgrid_resolution=meshgrid_resolution,
            )
            tunnel_r = ElipticalPolytunnel(
                semi_major_axis=semi_major_axis,
                length=length,
                tilt=tilt,
                azimuthal_orientation=azimuthal_orientation,
                x_shift=-3.0,
                meshgrid_resolution=meshgrid_resolution,
            )

            ground_grid = tunnel.generate_ground_grid()
            ground_grid_x, ground_grid_y, ground_grid_z = (
                ground_grid[0],
                ground_grid[1],
                ground_grid[2],
            )

            normals_unit_surface, areas_surface = tunnel.surface_element_unit_vectors()
            normals_unit_ground, areas_ground = tunnel.ground_element_unit_vectors()
            tilts_unit = tunnel.surface_tilt(normals_unit_surface)

            surface_grid, solar_cells = tunnel.generate_surface()
            surface_grid_x, surface_grid_y, surface_grid_z = (
                surface_grid[0],
                surface_grid[1],
                surface_grid[2],
            )

            distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(
                ground_grid, surface_grid
            )
    except Exception:
        print(FAILED)
    else:
        print(DONE)
        print(f"Geometry calculation: {geometry_timer()} seconds")

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


if __name__ == "__main__":
    main(sys.argv[1:])
