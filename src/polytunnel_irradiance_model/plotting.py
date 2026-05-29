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

import enum
import functools

from typing import Iterable

import matplotlib.animation as animation
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .__utils__ import spectrum_to_flux, spectrum_to_par
from .polytunnel import Polytunnel

__all__ = (
    "plot_animation",
    "plot_spectrum",
    "SpectralUnits",
)

# MM:
#   Conversion factor from mm to inches.
MM: float = 1 / 25.4


class SpectralUnits(enum.Enum):
    """
    Contains information about the units of the plot being requested.

    - PAR_FLUX:
        Will plot spectral flux, in photons, in the PAR region.

    - PAR_IRRADIANCE:
        Will plot irradiance, in W/m^2, in the PAR region.

    - PHOTON_FLUX:
        Will plot spectral flux, in photons, across the whole wavelength range provided.

    - IRRADIANCE:
        Will plot irradiance, in W/m^2, across the whole wavelength range provided.

    """

    IRRADIANCE = "Irradiance ($G$) / W/m$^{2}$"
    PAR_FLUX = r"PAR flux ($\Phi_{\rm{PAR}}$) / $\mu$mol/cm$^2$"
    PAR_IRRADIANCE = "PAR irradiance ($G_{\rm{PAR}}$) / W/m$^{2}$"
    PHOTON_FLUX = r"Photon flux ($\Phi_{\gamma}$) / $\mu$mol/cm$^2$"


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


def plot_spectrum(
    data: (
        list[tuple[np.ndarray | pd.Series, str]]
        | list[tuple[np.ndarray | pd.Series, str, float]]
    ),
    wavelength_range: Iterable[float],
    *,
    index: int = 0,
    plotting_wavelength_range: Iterable[float] | None = None,
    show: bool = False,
    small: bool = False,
    spectral_units: SpectralUnits = SpectralUnits.IRRADIANCE,
    title: str = "animation",
) -> plt.Figure:
    """
    Plot a spectrum/spectra in either flux or irradiance units.

    :param: data:
        The data to plot. This should come as tuples containing, pairwise:
        - the spectrum to plot,
        - a label to use for the spectrum,
        - (optional) the colour index to use.

    :param: wavelength_range:
        The wavelength range to over which the spectra are defined be summed. By
        default, this range is also the range over which values should be summed.

    :param: index:
        A variable to use for naming plots.

    :param: plotting_wavelength_range:
        The wavelength range to use for plotting, if different from the default
        provided over which the spectra are defined.

    :param: show:
        Whether to show plots (`True`) or not (`False`).

    :param: small:
        Whether to plot a small figure (`True`) or a large figure (`False`).

    :param: spectral_units:
        The spectral units to use for plotting.

    :param: title:
        The tiel to use for the plots.

    """

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
        import matplotlib.pyplot as plt

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

    # Helper function for keeping track of colours if not specified and extracting the
    # correct index if specified.
    current_colour_index: int = 0

    def _colour_index(
        data_entry: (
            tuple[np.ndarray | pd.Series, str]
            | tuple[np.ndarray | pd.Series, str, float]
        ),
    ) -> int:
        """
        Return the colour index.

        :param: data_entry:
            The spectral data to investigate.

        :returns:
            The colour index to use.

        """

        try:
            return data_entry[2]
        except IndexError:
            current_colour_index += 1
            return current_colour_index

    # Code for determining the units of the spectrum to use.
    if plotting_wavelength_range is None:
        plotting_wavelength_range = wavelength_range

    if SpectralUnits == SpectralUnits.PHOTON_FLUX:
        spectral_function: callable = spectrum_to_flux
    elif SpectralUnits == SpectralUnits.PAR_FLUX:
        spectral_function = functools.partial(
            spectrum_to_par, par_wavelength_series=plotting_wavelength_range
        )
    else:
        spectral_function = lambda x: x

    # Plot the data, pairwise.
    fig = plt.figure(figsize=(171 * MM, 120 * MM) if not small else (83 * MM, 60 * MM))

    for entry in data:
        plt.plot(
            plotting_wavelength_range,
            spectral_function(entry[0]),
            label=entry[1],
            color=f"C{_colour_index(entry)}",
        )

    plt.xlabel("Wavelength / nm")
    plt.ylabel(spectral_units.value)
    plt.legend(loc="upper right", fontsize=7)

    (axis := plt.gca()).tick_params(axis="both", which="major", labelsize=7)

    plt.savefig(
        f"{title}_{index}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05
    )
    plt.savefig(
        f"{title}_{index}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    if show:
        plt.show()

    return fig
