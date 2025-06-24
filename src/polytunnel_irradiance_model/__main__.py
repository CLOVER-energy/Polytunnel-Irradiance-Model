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
import os
import sys
import time

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import visualisation as viz
import yaml

from functions import *
from geometry import Polytunnel
from sun import Sun
from irradiance import TunnelIrradiance
from tracing import Tracing
from typing import Any


import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module=r".*vectorized_tmm_dispersive_multistack.*"
)

__all__ = ("compute_surface_grid", "main")


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
        "--length",
        "-l",
        type=float,
        default=5,
        help="The length, in metres, of the polytunnel segment to model.",
    )
    polytunnel_arguments.add_argument(
        "--polytunnel-type",
        "-pt",
        type=int,
        choices={0, 1},
        default=0,
        help="The type of polytunnel being modelled:\n"
        "0 - A circular cross section\n"
        "1 - An eliptical cross section.",
    )
    polytunnel_arguments.add_argument(
        "--cell-spacing",
        type=float,
        default=1,
        help="The spacing, in meters, between modules on the polytunnel surface.",
    )
    polytunnel_arguments.add_argument(
        "--tilt",
        type=float,
        default=0.0,
        help="The angle by which the central axis of the polytunnel is tilted, in "
        "degrees.",
    )
    polytunnel_arguments.add_argument(
        "--azimuthal_orientation",
        type=float,
        default=0.0,
        help="The orientation, relative to North--South, of the central polytunnel "
        "axis.",
    )
    polytunnel_arguments.add_argument(
        "--transmissivity",
        "-tau",
        type=float,
        default=1,
        help="The transmissivity of the polytunnel material; i.e., the plastic in the "
        "region where no PV modules are present.",
    )

    # Eliptical polytunnel arguments
    eliptical_arguments = parser.add_argument_group(
        "eliptical arguments",
        description="Arguments to use only when specifying an elipitical geometry.",
    )
    eliptical_arguments.add_argument(
        "--semi-major-axis",
        "--radius-a",
        "-smaj-ax",
        type=float,
        default=8,
        help="The length of the longer axis in the elipse. Default value of 8 m.",
    )
    eliptical_arguments.add_argument(
        "--semi-minor-axis",
        "--radius-b",
        "-smin-ax",
        type=float,
        default=8,
        help="The length of the shorter axis in the elipse. Default value of 8 m.",
    )

    # Solar cell arguments.
    solar_cell_arguments = parser.add_argument_group(
        "solar-cell arguments",
        description="Arguments for specifying the configuration of the PV cells/modules.",
    )
    solar_cell_arguments.add_argument(
        "--sloar-cell-file",
        "--materials-file",
        type=str,
        default="solar_cell.yaml",
        help="The path to the materials inputs file.",
    )
    # parser.add_argument(
    #     "--initial_cell_spacing",
    #     type=float,
    #     default=0.0,
    #     help="initial_cell_spacing (default: 1.0 (1 meter)                  )",
    # )

    parser.add_argument(
        "--res-meshgrid",
        type=float,
        default=1.0,
        help="The resolution of the mesh grid; default of 1 m.",
    )

    return parser.parse_args(args)


def main(args: list[Any]) -> None:
    """
    Main function for operating the Polytunnel-Irradiance-Model.

    :param: args:
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = parse_args(args)

    # Open the material information.
    with open(parse_args.material_file, "r", encoding="UTF-8") as material_file:
        material_information = yaml.safe_load(material_file)

    import pdb

    pdb.set_trace()

    date_simulation = params["start_time_str"][:10].replace("-", "")
    simulation_data = f"{date_simulation}_stack_{params['multistack']}_r1_{params['radius1']}_r2_{params['radius2']}"
    simulation_data += f"_donor_{material_list[2]}_acceptor_{material_list[3]}_cell_thickness_{params['cell_thickness']}"
    simulation_data += f"_cell_spacing_{params['cell_spacing']}_resolution_{params['res_meshgrid']}_ptunnelLength_{params['length']}"

    if not os.path.exists("Matched_Dates"):
        os.makedirs("Matched_Dates")

    if not os.path.exists("Logs_Dates"):
        os.makedirs("Logs_Dates")

    memory_profile_log = f"Logs_Dates/memory_profile_{simulation_data}.log"

    # Configure memory profiler stream before running main(**params)
    with open(memory_profile_log, "w+") as f:
        # assign decorator
        profiler_stream = f
        main(**params)

    # Obtain date (Format): YYYYMMDD
    # Default date_simulation == '20240730', 30th July 2024
    date_simulation = start_time_str[:10].replace("-", "")
    parameters_sim = f"{date_simulation}_stack_{multistack}_r1_{radius1}_r2_{radius2}_donor_{material_list[2]}_acceptor_{material_list[3]}_cell_thickness_{cell_thickness}_cell_spacing_{cell_spacing}_resolution_{res_meshgrid}_ptunnelLength_{length}"

    memory_information = {}

    if not os.path.isdir(f"figures/Date_{date_simulation}"):
        os.makedirs(f"figures/Date_{date_simulation}")

    start_simulation_time = time.time()

    # tunnel geometry#
    tunnel = Polytunnel(
        radius1=radius1,
        length=length,
        tilt=tilt,
        azimuthal_orientation=azimuthal_orientation,
        theta_margin=0,
        cell_thickness=cell_thickness,
        cell_spacing=cell_spacing,
        radius2=radius2,
        res_meshgrid=res_meshgrid,
        initial_cell_spacing=initial_cell_spacing,
    )
    tunnel_l = Polytunnel(
        radius1=radius1,
        length=length,
        tilt=tilt,
        azimuthal_orientation=azimuthal_orientation,
        x_shift=3.0,
        res_meshgrid=res_meshgrid,
    )
    tunnel_r = Polytunnel(
        radius1=radius1,
        length=length,
        tilt=tilt,
        azimuthal_orientation=azimuthal_orientation,
        x_shift=-3.0,
        res_meshgrid=res_meshgrid,
    )

    ground_grid = tunnel.generate_ground_grid()
    ground_grid_x, ground_grid_y, ground_grid_z = (
        ground_grid[0],
        ground_grid[1],
        ground_grid[2],
    )

    memory_information = dictionary_update(
        memory_information, "ground_grid_x_bytes", ground_grid_x
    )
    memory_information = dictionary_update(
        memory_information, "ground_grid_y_bytes", ground_grid_y
    )
    memory_information = dictionary_update(
        memory_information, "ground_grid_z_bytes", ground_grid_z
    )
    memory_information = dictionary_update(
        memory_information, "ground_grid_all_bytes", ground_grid
    )

    normals_unit_surface, areas_surface = tunnel.surface_element_unit_vectors()
    normals_unit_ground, areas_ground = tunnel.ground_element_unit_vectors()
    tilts_unit = tunnel.surface_tilt(normals_unit_surface)

    memory_information = dictionary_update(
        memory_information, "normals_unit_surface_bytes", normals_unit_surface
    )
    memory_information = dictionary_update(
        memory_information, "areas_surface_bytes", areas_surface
    )
    memory_information = dictionary_update(
        memory_information, "normals_unit_ground_bytes", normals_unit_ground
    )
    memory_information = dictionary_update(
        memory_information, "areas_ground_bytes", areas_ground
    )
    memory_information = dictionary_update(
        memory_information, "tilts_unit_bytes", tilts_unit
    )

    surface_grid, solar_cells = tunnel.generate_surface()
    surface_grid_x, surface_grid_y, surface_grid_z = (
        surface_grid[0],
        surface_grid[1],
        surface_grid[2],
    )

    distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(
        ground_grid, surface_grid
    )

    memory_information = dictionary_update(
        memory_information, "surface_grid_x_bytes", surface_grid_x
    )
    memory_information = dictionary_update(
        memory_information, "surface_grid_y_bytes", surface_grid_y
    )
    memory_information = dictionary_update(
        memory_information, "surface_grid_z_bytes", surface_grid_z
    )
    memory_information = dictionary_update(
        memory_information, "surface_grid_all_bytes", surface_grid
    )
    memory_information = dictionary_update(
        memory_information, "solar_cells_bytes", solar_cells
    )
    memory_information = dictionary_update(
        memory_information, "distance_grid_bytes", distance_grid
    )
    memory_information = dictionary_update(
        memory_information,
        "separation_unit_vector_grid_bytes",
        separation_unit_vector_grid,
    )

    end_geometry_time = time.time()
    print(
        f"Geometry calculation time: {end_geometry_time - start_simulation_time } seconds"
    )

    d = 2 * radius1
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

    memory_information = dictionary_update(
        memory_information, "time_array_bytes", time_array
    )
    memory_information = dictionary_update(
        memory_information, "altitude_array_bytes", altitude_array
    )
    memory_information = dictionary_update(
        memory_information, "azimuth_array_bytes", azimuth_array
    )

    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)
    sun_surface_grid, sun_incident = sun.sunvec_tilts_grid(
        sun_vecs, normals_unit_surface
    )

    memory_information = dictionary_update(
        memory_information, "sun_surface_grid_bytes", sun_surface_grid
    )
    memory_information = dictionary_update(
        memory_information, "sun_incident_bytes", sun_incident
    )
    memory_information = dictionary_update(
        memory_information, "sun_vecs_bytes", sun_vecs
    )
    memory_information = dictionary_update(
        memory_information, "sun_positions_bytes", sun_positions
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
    tracer = Tracing(tunnel, sun_vecs, surface_grid, tilts_unit, radius1, radius2)
    gradient_grid, angle_grid, surface_gradient_grid, surface_angle_grid = (
        tracer.find_tangent_gradient()
    )

    memory_information = dictionary_update(
        memory_information, "gradient_grid_bytes", gradient_grid
    )
    memory_information = dictionary_update(
        memory_information, "angle_grid_bytes", angle_grid
    )
    memory_information = dictionary_update(
        memory_information, "surface_gradient_grid_bytes", surface_gradient_grid
    )
    memory_information = dictionary_update(
        memory_information, "surface_angle_grid_bytes", surface_angle_grid
    )

    shading_calculation_time = time.time()
    print(
        f"Shading calculation time: { shading_calculation_time - sun_positions_calculation_time } seconds"
    )

    irradiance = TunnelIrradiance(tunnel, radius1, radius2, length)
    shaded_exposure_map = irradiance.shading_exposure_map(
        angle_grid, surface_angle_grid, sun_vecs
    )

    # zero tilt, PAR region: 400-700
    # optical_wavelengths in nm
    # optical intensities in W/m^2/nm
    # spectra_frames units in W/m^2
    optical_wavelengths, optical_intensities, spectra_frames = sun.get_spectra(400, 700)

    memory_information = dictionary_update(
        memory_information, "optical_wavelengths_bytes", optical_wavelengths
    )
    memory_information = dictionary_update(
        memory_information, "optical_intensities_bytes", optical_intensities
    )
    memory_information = dictionary_update(
        memory_information, "spectra_frames_bytes", spectra_frames
    )

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

    memory_information = dictionary_update(
        memory_information, "complex_array_bytes", complex_array
    )
    memory_information = dictionary_update(
        memory_information, "t_grid_frames_bytes", t_grid_frames
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

    memory_information = dictionary_update(
        memory_information, "solar_cell_spectra_bytes", solar_cell_spectra
    )
    memory_information = dictionary_update(
        memory_information, "solar_cell_spectra_shaded_bytes", solar_cell_spectra_shaded
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

    memory_information = dictionary_update(
        memory_information, "transmitted_frames_bytes", transmitted_frames
    )
    memory_information = dictionary_update(
        memory_information, "absorbed_frames_bytes", absorbed_frames
    )
    memory_information = dictionary_update(
        memory_information, "transmitted_int_bytes", transmitted_int
    )
    memory_information = dictionary_update(
        memory_information, "absorbed_int_bytes", absorbed_int
    )

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

    memory_information = dictionary_update(
        memory_information,
        "direct_ground_irradiance_frames_bytes",
        direct_ground_irradiance_frames,
    )
    memory_information = dictionary_update(
        memory_information, "direct_ground_int_bytes", direct_ground_int
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

    memory_information = dictionary_update(
        memory_information,
        "diffuse_ground_irradiance_frames_bytes",
        diffuse_ground_irradiance_frames,
    )
    memory_information = dictionary_update(
        memory_information, "diffuse_ground_int_bytes", diffuse_ground_int
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

    memory_information = dictionary_update(
        memory_information,
        "global_ground_irradiance_frames_bytes",
        global_ground_irradiance_frames,
    )
    memory_information = dictionary_update(
        memory_information, "global_ground_int_bytes", global_ground_int
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

    memory_information = dictionary_update(
        memory_information,
        "direct_photon_spectra_frames_bytes",
        direct_photon_spectra_frames,
    )
    memory_information = dictionary_update(
        memory_information, "direct_photon_par_spectra_bytes", direct_photon_par_spectra
    )
    memory_information = dictionary_update(
        memory_information, "direct_photon_par_bytes", direct_photon_par
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

    memory_information = dictionary_update(
        memory_information,
        "diffuse_photon_spectra_frames_bytes",
        diffuse_photon_spectra_frames,
    )
    memory_information = dictionary_update(
        memory_information,
        "diffuse_photon_par_spectra_bytes",
        diffuse_photon_par_spectra,
    )
    memory_information = dictionary_update(
        memory_information, "diffuse_photon_par_bytes", diffuse_photon_par
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

    memory_information = dictionary_update(
        memory_information,
        "global_photon_spectra_frames_bytes",
        global_photon_spectra_frames,
    )
    memory_information = dictionary_update(
        memory_information, "global_photon_par_spectra_bytes", global_photon_par_spectra
    )
    memory_information = dictionary_update(
        memory_information, "global_photon_par_bytes", global_photon_par
    )

    par_global_calculation_time = time.time()
    print(
        f"PAR Global calculation time: { par_global_calculation_time - par_diffuse_calculation_time  } seconds"
    )

    info_memory_df = pd.DataFrame(
        {"Variable": memory_information.keys(), "Bytes": memory_information.values()}
    )
    path_memory = (
        f"figures/Date_{date_simulation}/memory_information_bytes_{parameters_sim}.csv"
    )
    info_memory_df.to_csv(path_memory, index=False)

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
