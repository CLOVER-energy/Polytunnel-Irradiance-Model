import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import timedelta
from scipy.integrate import trapezoid

def round_10_min(dt):

    # Find the number of minutes past the hour
    minutes = dt.minute
    
    # Calculate the rounding direction
    if minutes % 10 >= 5:

        # Round up
        rounded_minute = (minutes // 10 + 1) * 10

    else:

        # Round down
        rounded_minute = (minutes // 10) * 10
    
    # Handle overflow to the next hour
    if rounded_minute == 60:

        dt = dt.replace(minute=0) + timedelta(hours=1)

    else:

        dt = dt.replace(minute=rounded_minute)
    
    # Zero out seconds and microseconds
    dt = dt.replace(second=0, microsecond=0)
    
    return dt

def plot_sun(time_array, altitude_array, azimuth_array, irradiance_frames, filename):

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_array, altitude_array, label='Altitude', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Altitude (degrees)')
    plt.title('Solar Altitude Over Time')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time_array, azimuth_array, label='Azimuth', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Azimuth (degrees)')
    plt.title('Solar Azimuth Over Time')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_array, irradiance_frames, label='Irradiance', color='red')
    plt.xlabel('Time')
    plt.ylabel(r"Irradiance $\left(\frac{W}{m^{2}}\right)$")
    plt.title('Solar Irradiance Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close()

def set_axes_equal(ax):
    ''' 
    Set the 3D plot axes to have equal scale so that spheres appear as spheres,
    and cubes as cubes, etc. This is one possible solution to Matplotlib's lack
    of a set_aspect method for 3D plots.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = np.array([x_range, y_range, z_range]).max()

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
    ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
    ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

def plot_surface(X, Y, Z, normals_unit, X_ground, Y_ground, Z_ground, normals_unit_ground, sun_vec):
    # Normal vectors at the center points
    U = normals_unit[0]
    V = normals_unit[1]
    W = normals_unit[2]
    U_ground = normals_unit_ground[0]
    V_ground = normals_unit_ground[1]
    W_ground = normals_unit_ground[2]
    sun_dir_x = sun_vec[0]
    sun_dir_y = sun_vec[1]
    sun_dir_z = sun_vec[2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surfaces with custom colors
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.4)  # Polytunnel surface
    ax.plot_surface(X_ground, Y_ground, Z_ground, color='brown', alpha=0.6)  # Ground surface

    # Plot normal vectors
    ax.quiver(X_ground, Y_ground, Z_ground, U_ground, V_ground, W_ground, length=1, color='green')
    ax.quiver(X, Y, Z, U, V, W, length=1, color='black')

    # Plot sun rays
    ax.quiver(X, Y, Z, sun_dir_x, sun_dir_y, sun_dir_z, color='red')

    # Remove grid and axes
    ax.grid(False)
    ax.axis('off')

    # Set equal aspect ratio for the axes
    set_axes_equal(ax)

    # plt.show()
    plt.close()

def plot_irradiance(ground_grid_x, ground_grid_y, irradiance_array):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Initialize the contour plot
    contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array, levels=100, cmap='hot')
    cbar = fig.colorbar(contour, ax=ax, label='Irradiance')

    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_title('Irradiance Pattern on the Polytunnel surface')
    # plt.show()
    plt.close()

def animate_irradiance(time_array, ground_grid_x, ground_grid_y, irradiance_array, label_colorbar, filename):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    irradiance_array = np.array(irradiance_array, dtype=np.float64)
    # irradiance_array = np.array(irradiance_array, dtype=object)

    irradiance_array = np.nan_to_num(irradiance_array, nan=0.0, posinf=0.0, neginf=0.0)

   # Find the global min and max of the irradiance array
    irradiance_min = 0
    irradiance_max = np.max(irradiance_array)
    # irradiance_max = np.max([max(item.all()) for item in irradiance_array.flatten()])
    list_index = next(i for i, sublist in enumerate(irradiance_array) if irradiance_max in sublist)

    # Initialize the contour plot with the first frame, using the global min and max for color limits
    contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[list_index], levels=100, cmap='hot', vmin=irradiance_min, vmax=irradiance_max)
    cbar = fig.colorbar(contour, ax=ax, label=label_colorbar)
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')


    def update(frame):
        ax.clear()
        contour = ax.contourf(ground_grid_x, ground_grid_y, irradiance_array[frame], levels=100, cmap='hot', vmin=irradiance_min, vmax=irradiance_max)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        ax.set_title(f'Irradiance Grid - Time {round_10_min(time_array[frame])}')
        return contour,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(irradiance_array), interval=50, blit=True)

    # Save the animation
    anim.save(filename, writer='ffmpeg')

    # plt.show()
    plt.close()

def plot_power(time_array, power_ground_diffuse, power_ground_direct, power_ground, filename):
   
    plt.figure(figsize=(10, 7))
    # Plot the stacked area plot
    plt.stackplot(time_array, power_ground_direct, power_ground_diffuse, labels=['Ground (Direct)', 'Ground (Diffuse)'])
    # Overlay the global power plot
    plt.plot(time_array, power_ground, label='Ground (Global)', color='black', linestyle='--')

    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.savefig(filename)
    # plt.show()
    plt.close()

def plot_coverage(surface_grid, visible_solid_angle):
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(surface_grid[0], surface_grid[1], visible_solid_angle, levels=100, cmap='viridis')
    plt.colorbar(label='Visible Solid Angle (steradians)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visible Sky Coverage on Central Polytunnel')
    # plt.show()
    plt.close()

def tmm_grid(optical_wavelengths, Theta, t_amp, filename=None):
    
    plt.figure(figsize=(8, 6))

    # Create a heatmap using pcolormesh
    plt.pcolormesh(optical_wavelengths, Theta, t_amp, cmap='viridis', shading='auto', vmin=0, vmax=np.max(t_amp))

    # Add a color bar to show the transmission amplitude scale (0 to 1)
    cbar = plt.colorbar()
    cbar.set_label('Transmission Amplitude')

    # Label the axes
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Tilt Angle (Radians)')

    # Set a title for the plot
    plt.title('TMM of Transmission Amplitudes for Tilt Angles and Wavelengths')

    if filename:
        plt.savefig(filename, dpi=300)
    else:
        plt.savefig('figures/tmm-transmission.png', dpi=300)

    # Display the plot
    # plt.show()
    plt.close()
    
def every_grid_plot(spectra_frames, time_array, optical_wavelengths, res_minutes, filename):
    
    # Number of time Frames
    n_frames = spectra_frames.shape[0]
    # Label for every frame: HH:MM
    times_labels = [f'{item.hour}:{item.minute:02}' for item in time_array]

    # Colors for timeframe colorbar
    # colorID = ['blue', 'yellow']
    colorID = ['navy', 'lime']
    cmap = LinearSegmentedColormap.from_list('my_colormap', colorID)
    norm = Normalize(vmin=0, vmax=n_frames - 1)  # 
    colors = [cmap(norm(i)) for i in range(n_frames)]  # 
    
    # number of grids from sides of Polytunnel, same as n_points_angular, parameters file
    sides_tiles = spectra_frames.shape[1]
    # number of grids from length of Polytunnel, same as n_points, parameters file
    length_tiles = spectra_frames.shape[2]

    # Size of figure, shape (n_points_angular, n_points)
    fig, ax = plt.subplots(sides_tiles, length_tiles, figsize=( 4 * length_tiles, 4 * sides_tiles ))
    max_val = 0

    # region size    
    region_size = sides_tiles // 3
    
    # left side
    upper_limit = region_size              
    # right side
    lower_start = sides_tiles - region_size      

    # Plot every region of grid 
    aspectra_values_mean = []
    cspectra_values_mean = []
    lspectra_values_mean = []
    rspectra_values_mean = []

    for i in range(n_frames):
        aintegral = 0
        cintegral, cval_denominator = 0, 0
        lintegral, lval_denominator = 0, 0
        rintegral, rval_denominator = 0, 0
        
        for q in range(sides_tiles):  # 

            for p in range(length_tiles):  #
                value = trapezoid(spectra_frames[i][q][p], optical_wavelengths, 0.01) 
                aintegral += value

                if q < upper_limit:
                    lintegral += value
                    lval_denominator += 1

                elif q >= lower_start:
                    rintegral += value 
                    rval_denominator += 1 
                else:
                    cintegral += value
                    cval_denominator += 1


                ax[q, p].plot(optical_wavelengths, spectra_frames[i][q][p], color=colors[i])  # Curva de cada frame
                if max(spectra_frames[i][q][p])> max_val:
                    max_val = max(spectra_frames[i][q][p])
        aspectra_values_mean.append( aintegral / ( sides_tiles * length_tiles ) )
        cspectra_values_mean.append( cintegral / cval_denominator )
        lspectra_values_mean.append( lintegral / lval_denominator )
        rspectra_values_mean.append( rintegral / rval_denominator )

    
    for q in range(sides_tiles):  # 
        for p in range(length_tiles):  # 
            ax[q,p].set_ylim([0, 1.05 * max_val])
        
    # ColorBar: TOP
    sm = ScalarMappable(cmap=cmap, norm=norm)  # Create mappable object 
    cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.03])  # Position of colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')  # Add Colorbar

    # time labels for colorbar
    cticks = [item * (60//res_minutes)*3 for item in range(8) ] + [len(time_array) -1 ]
    cbarlabel = [times_labels[int(item)] for item in cticks]
    cbar.set_ticks(cticks)
    cbar.set_ticklabels(cbarlabel, fontsize = 26)
    
    # Save Figure
    # plt.savefig(filename, dpi = 300)
    plt.savefig(filename)
    plt.close()
    return aspectra_values_mean, cspectra_values_mean, lspectra_values_mean, rspectra_values_mean



