# main.py
import numpy as np
import matplotlib.pyplot as plt
from geometry import Polytunnel
from sun import Sun
from irradiance import TunnelIrradiance
import visualisation as viz
from tracing import Tracing
import tmm as tmm
import tmm_fast as vectmm
import torch

__all__ = ("compute_surface_grid",)

def compute_surface_grid():
    """Returns the surface grid"""

def main(start_time_str='2024-07-30T00:00:00Z', end_time_str='2024-07-30T23:59:59Z', latitude=51.1950, longitude=0.2757, res_minutes = 1, n_points = 1000, length = 20, radius = 5, xy_angle = 0, z_angle = 0, transmissivity = 1):
    
    #tunnel geometry#
    tunnel = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, theta_margin = 0, cell_thickness=2, cell_gap=2)
    tunnel_l = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, x_shift = 3.0)
    tunnel_r = Polytunnel(radius=radius, length=length, n_points=n_points, xy_angle=xy_angle, z_angle=z_angle, x_shift = -3.0)

    ground_grid = tunnel.generate_ground_grid()
    ground_grid_x, ground_grid_y, ground_grid_z = (ground_grid[0], ground_grid[1], ground_grid[2])

    normals_unit_surface, areas_surface = tunnel.surface_element_unit_vectors()
    normals_unit_ground, areas_ground = tunnel.ground_element_unit_vectors()
    tilts_unit = tunnel.surface_tilt(normals_unit_surface)

    surface_grid, solar_cells = tunnel.generate_surface()
    surface_grid_x, surface_grid_y, surface_grid_z = (surface_grid[0], surface_grid[1], surface_grid[2])
    
    distance_grid, separation_unit_vector_grid = tunnel.generate_distances_grid(ground_grid, surface_grid)

    d = 2*radius

    #sun transits#
    sun = Sun(start_time=start_time_str, end_time=end_time_str, latitude=latitude, longitude=longitude, resolution_minutes=res_minutes)
    altitude_array, azimuth_array = sun.generate_sun_positions()
    time_array = sun.get_times()
    sun_positions = list(zip(altitude_array, azimuth_array))
    sun_vecs = sun.generate_sun_vecs(sun_positions)
    sun_surface_grid, sun_incident = sun.sunvec_tilts_grid(sun_vecs, normals_unit_surface)

    #shading#
    tracer = Tracing(tunnel, sun_vecs, surface_grid, tilts_unit, d, radius)
    gradient_grid, angle_grid, surface_gradient_grid, surface_angle_grid = tracer.find_tangent_gradient()
    solid_angle_grid = tracer.solid_angle_grid(angle_grid, surface_angle_grid)

    irradiance = TunnelIrradiance(tunnel, radius, length)
    shaded_exposure_map = irradiance.shading_exposure_map(angle_grid, surface_angle_grid, sun_vecs)
    optical_wavelengths, optical_intensities, spectra_frames = sun.get_spectra(300, 700) #zero tilt

    #TMM#
    material_list = ['ag', 'moo3', 'pce10:pc61bm', 'zno', 'moo3', 'pdbt[2f]t:pc71bm', 'zno', 'ito']
    d_list = np.array(['inf', 30, 7, 70, 35, 15, 100, 25, 110, 'inf'])
    complex_array = tracer.n_list_wavelength(material_list, optical_wavelengths)
    t_grid_frames = irradiance.t_grid(sun_incident, optical_wavelengths, complex_array, d_list)

    #Solar Spectra#
    solar_cell_spectra = irradiance.solar_cells_irradiance_rays(optical_intensities, t_grid_frames, solar_cells)
    solar_cell_spectra_shaded = irradiance.shaded_irradiance_spectra(solar_cell_spectra, shaded_exposure_map)

    #Irradiance#
    transmitted_frames, absorbed_frames = irradiance.transmitted_absorbed_spectra(optical_wavelengths, solar_cell_spectra_shaded, sun_surface_grid)
    transmitted_int, absorbed_int = irradiance.int_spectra(optical_wavelengths, transmitted_frames), irradiance.int_spectra(optical_wavelengths, absorbed_frames)

    #Direct Irradiance (ground)#
    direct_ground_irradiance_frames = irradiance.direct_irradiance_spectra_ground(ground_grid, normals_unit_ground, surface_grid, distance_grid, sun_vecs, transmitted_frames)
    direct_ground_int = irradiance.int_spectra(optical_wavelengths, direct_ground_irradiance_frames)

    #Diffuse Irradiance (ground)#
    diffuse_ground_irradiance_frames = irradiance.diffuse_irradiance_spectra_ground(distance_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, areas_surface, absorbed_frames)
    diffuse_ground_int = irradiance.int_spectra(optical_wavelengths, diffuse_ground_irradiance_frames)

    #Global Irradiance (ground)#
    global_ground_irradiance_frames = irradiance.global_irradiance_spectra_ground(diffuse_ground_irradiance_frames, direct_ground_irradiance_frames)
    global_ground_int = irradiance.int_spectra(optical_wavelengths, global_ground_irradiance_frames)

    #Photon spectra PAR#
    photon_spectra_frames = irradiance.power_to_photon_spectra(optical_wavelengths, global_ground_irradiance_frames)
    photon_par_spectra = irradiance.par_spectra(optical_wavelengths, photon_spectra_frames)
    photon_par = irradiance.int_spectra(photon_par_spectra)

    #diffuse_irradiance_frames = irradiance.diffuse_irradiance_ground(distance_grid, separation_unit_vector_grid, normals_unit_ground, normals_unit_surface, areas_surface, irradiance_frames, transmissivity)
    #direct_irradiance_frames = irradiance.direct_irradiance_ground(normals_unit_ground, sun_vecs, spectra_frames, transmissivity)

    #power_total_ground_diffuse = irradiance.power(areas_ground, diffuse_irradiance_frames)
    #power_total_surface = irradiance.power(areas_surface, irradiance_frames)
    #power_total_ground_direct = irradiance.power(areas_ground, direct_irradiance_frames)
    #power_total_out = np.array(power_total_ground_diffuse) + np.array(power_total_ground_direct)

    viz.plot_sun(time_array, altitude_array, azimuth_array, spectra_frames, "figures/sun.png")

    viz.plot_surface(surface_grid_x, surface_grid_y, surface_grid_z, normals_unit_surface, ground_grid_x, ground_grid_y, ground_grid_z, normals_unit_ground, sun_vec=sun_vecs[72])

    #viz.plot_irradiance(surface_grid_x, surface_grid_y, irradiance_frames[60])

    viz.animate_irradiance(time_array, surface_grid_x, surface_grid_y, transmitted_int, "figures/direct-irradiance-surface-animation.mp4")

    viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, direct_ground_int, "figures/direct-irradiance-ground-animation.mp4")
    
    viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, diffuse_ground_int, "figures/diffuse-irradiance-ground-animation.mp4")

    viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, global_ground_int, "figures/global-irradiance-ground-animation.mp4")

    viz.animate_irradiance(time_array, ground_grid_x, ground_grid_y, photon_par, "figures/photon-par-animation.mp4")

    #viz.plot_power(time_array, power_total_ground_diffuse, power_total_ground_direct, power_total_out, 'figures/power-received.png')
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No command-line arguments, use default values
        main()
    elif len(sys.argv) == 12:
        # Command-line arguments provided
        start_time_str = sys.argv[1]
        end_time_str = sys.argv[2]
        latitude = float(sys.argv[3])
        longitude = float(sys.argv[4])
        res_minutes = float(sys.argv[5])
        n_points = float(sys.argv[6])
        length = float(sys.argv[7])
        radius = float(sys.argv[8])
        xy_angle = float(sys.argv[9])
        z_angle = float(sys.argv[10])
        transmissivity = float(sys.argv[11])

        main(start_time_str, end_time_str, latitude, longitude, res_minutes, n_points, length, radius, xy_angle, z_angle, transmissivity)
    else:
        print("Usage: python main.py [<start_time> <end_time> <latitude> <longitude>]")
        print("Default values will be used if no arguments are provided.")
        sys.exit(1)
