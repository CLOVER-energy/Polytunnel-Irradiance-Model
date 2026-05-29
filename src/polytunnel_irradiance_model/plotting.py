########################################################################################
# plotting.py --- Plotting code for the polytunnel-irradiance model..                  #
#                                                                                      #
# Author(s): Benedict Winchester                                                       #
# Date created: Spring 2026                                                            #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `plotting.py`

The model functions to compute, utilising spectral ray-tracing tools, the irradiance
distribution within a curved structure, _e.g._, a polytunnel.

"""

import functools

from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np

from .__utils__ import spectrum_to_flux, spectrum_to_par
from .polytunnel import Polytunnel

__all__ = ("plot_animation",)

# MM:
#   Conversion factor from mm to inches.
MM: float = 1 / 25.4


def plot_animation(
    data: np.ndarray,
    polytunnel: Polytunnel,
    wavelength_range: Iterable[float],
    *,
    index: int = 0,
    modelling_temporal_resolution: int = 60,
    plotting_wavelength_range: Iterable[float] | None = None,
    show: bool = False,
    title: str = "animation",
) -> animation.FuncAnimation:
    """
    Plots an animation based on the input information.

    :param: data:
        The data to plot.

    :param: polytunnel:
        The polytunnel to plot

    :param: wavelength_range:
        The wavelength range to over which the spectra are defined be summed. By
        default, this range is also the range over which values should be summed.

    :param: index:
        A variable to use for naming plots.

    :param: modelling_temporal_resolution:
        The temporal resolution used in the modeling.

    :param: plotting_wavelength_range:
        The wavelength range to use for plotting, if different from the default
        provided over which the spectra are defined.

    :param: show:
        Whether to show plots (`True`) or not (`False`).

    :param: title:
        The tiel to use for the plots.

    """

    if plotting_wavelength_range is None:
        plotting_wavelength_range = wavelength_range
        spectral_function: callable = spectrum_to_flux
        label_kwarg: str = "Photon"

    else:
        spectral_function = functools.partial(
            spectrum_to_par, par_wavelength_series=plotting_wavelength_range
        )
        label_kwarg = "PAR"

    fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

    # Create initial heatmap with dummy data
    initial_data = np.reshape(
        spectral_function(data[0], wavelength_series=wavelength_range).sum(axis=1),
        (
            _dim_x := polytunnel.meshgrid_resolution,
            _dim_y := polytunnel.length_wise_meshgrid_resolution,
        ),
    )
    vmin = 0
    vmax = spectral_function(data, wavelength_series=wavelength_range).sum(axis=2).max()
    heatmap = sns.heatmap(
        initial_data,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=True,
        ax=ax,
        cbar_kws={"label": f"{label_kwarg} flux ($\Phi$) / $\mu$mol/cm$^2$"},
    )
    heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)

    _ten_minutes: int = int(_ten_minutes := (60 / modelling_temporal_resolution))

    def update(time_index: int):
        ax.clear()  # clear previous heatmap
        this_data = np.reshape(
            spectral_function(data[time_index], wavelength_series=wavelength_range).sum(
                axis=1
            ),
            (_dim_x, _dim_y),
        )
        sns.heatmap(this_data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
        ax.set_title(
            f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
        )

    plt.xlabel("Depth index", fontsize=7)
    plt.ylabel("Width index", fontsize=7)

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=data.shape[0],
        interval=300,
        repeat=False,
    )
    ani.save(f"{title}_{index}.gif", writer="pillow", fps=5)
    if show:
        plt.show()

    return ani
