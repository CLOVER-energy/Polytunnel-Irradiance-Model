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
import matplotlib.lines as mlines
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

# LINESTYLE_MAP:
#   Information about the linestyles for easy lookup.
LINESTYLE_MAP: dict[str, tuple[int, tuple[int]]] = {
    "-": (0, ()),
    "--": (0, (5, 5)),
    "-.": (0, (3, 5, 1, 5)),
    ":": (0, (1, 5)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "dashed": (0, (5, 5)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "densely dashed": (0, (5, 1)),
    "densely dotted": (0, (1, 1)),
    "dotted": (0, (1, 5)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "loosely dashed": (0, (5, 10)),
    "loosely dotted": (0, (1, 10)),
}


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

    IRRADIANCE = "Irradiance ($G$) / W/m$^{2}$-nm"
    PAR_FLUX = r"PAR flux ($\Phi_{\rm{PAR}}$) / $\mu$mol/cm$^2$-nm"
    PAR_IRRADIANCE = "PAR irradiance ($G_{\rm{PAR}}$) / W/m$^{2}$-nm"
    PHOTON_FLUX = r"Photon flux ($\Phi_{\gamma}$) / $\mu$mol/cm$^2$-nm"


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
        | list[tuple[np.ndarray | pd.Series, str, float, str]]
    ),
    wavelength_range: Iterable[float],
    *,
    index: int = 0,
    palette: list[str] | sns.palettes._ColorPalette | None = None,
    plotting_wavelength_range: Iterable[float] | None = None,
    right_axis_data: (
        list[tuple[np.ndarray | pd.Series, str]]
        | list[tuple[np.ndarray | pd.Series, str, float]]
        | list[tuple[np.ndarray | pd.Series, str, float, str]]
        | None
    ) = None,
    show: bool = False,
    small: bool = False,
    spectral_units: SpectralUnits = SpectralUnits.IRRADIANCE,
    title: str = "animation",
    unique_legend: bool = False,
) -> plt.Figure:
    """
    Plot a spectrum/spectra in either flux or irradiance units.

    :param: data:
        The data to plot. This should come as tuples containing, pairwise:
        - the spectrum to plot,
        - a label to use for the spectrum,
        - (optional) the colour index to use,
        - (optional) the dashes to use.

    :param: wavelength_range:
        The wavelength range to over which the spectra are defined be summed. By
        default, this range is also the range over which values should be summed.

    :param: index:
        A variable to use for naming plots.

    :param: palette:
        A color palette to use.

    :param: plotting_wavelength_range:
        The wavelength range to use for plotting, if different from the default
        provided over which the spectra are defined.

    :param: right_axis_data:
        Data to use for the right axis.

    :param: show:
        Whether to show plots (`True`) or not (`False`).

    :param: small:
        Whether to plot a small figure (`True`) or a large figure (`False`).

    :param: spectral_units:
        The spectral units to use for plotting.

    :param: title:
        The tiel to use for the plots.

    :param: unique_legend:
        If specified, a unique legend will be constructed.

    """

    if palette is None:
        palette = [
            "#423252",
            "#4A688B",
            "#779FB1",
            "#36C7B8",
            "#FBC412",
            "#e04606",
        ]
        # palette = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]

    try:
        sns.set_palette(
            palette,
        )
    except UnboundLocalError:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_palette(palette)

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
            nonlocal current_colour_index
            current_colour_index += 1
            return current_colour_index

    # Code for determining the units of the spectrum to use.
    if plotting_wavelength_range is None:
        plotting_wavelength_range = wavelength_range

    if spectral_units == SpectralUnits.PHOTON_FLUX:
        spectral_function: callable = functools.partial(
            spectrum_to_flux, wavelength_series=wavelength_range
        )
    elif spectral_units == SpectralUnits.PAR_FLUX:
        spectral_function = functools.partial(
            spectrum_to_par,
            par_wavelength_series=plotting_wavelength_range,
            wavelength_series=wavelength_range,
        )
    else:
        spectral_function = lambda x: x

    # Plot the data, pairwise.
    fig = plt.figure(figsize=(171 * MM, 120 * MM) if not small else (83 * MM, 60 * MM))

    for entry in data:
        if len(entry) < 4:
            plt.plot(
                plotting_wavelength_range,
                spectral_function(entry[0]),
                label=entry[1],
                color=f"C{_colour_index(entry)}",
            )
        else:
            plt.plot(
                plotting_wavelength_range,
                spectral_function(entry[0]),
                label=entry[1],
                color=f"C{_colour_index(entry)}",
                linestyle=(
                    LINESTYLE_MAP.get(entry[3], (0, ()))
                    if isinstance(entry[3], str)
                    else entry[3]
                ),
            )

    plt.xlabel("Wavelength / nm")
    plt.ylabel(spectral_units.value)
    plt.legend(loc="upper right", fontsize=7)

    (axis_left := plt.gca()).tick_params(axis="both", which="major", labelsize=7)

    if unique_legend:
        # Determine unique labels and only keep these.
        handles, labels = axis_left.get_legend_handles_labels()
        unique_handles_labels: list[tuple[mlines.Line2D]] = []
        unique_labels: set[str] = set()
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles_labels.append((handle, label))
                unique_labels.add(label)

        plt.legend().remove()
        plt.legend(
            [entry[0] for entry in unique_handles_labels],
            [entry[1] for entry in unique_handles_labels],
            fontsize=7,
        )

    if right_axis_data is not None:
        axis_right = axis_left.twinx()
        for entry in right_axis_data:
            if len(entry) < 4:
                axis_right.plot(
                    plotting_wavelength_range,
                    spectral_function(entry[0]),
                    label=entry[1],
                    color=f"C{_colour_index(entry)}",
                )
            else:
                axis_right.plot(
                    plotting_wavelength_range,
                    spectral_function(entry[0]),
                    label=entry[1],
                    color=f"C{_colour_index(entry)}",
                    linestyle=(
                        LINESTYLE_MAP.get(entry[3], (0, ()))
                        if isinstance(entry[3], str)
                        else entry[3]
                    ),
                )

        axis_right.set_ylabel("Direct-irradiance response / normalised units")
        axis_right.tick_params(axis="both", which="major", labelsize=7)

        # Combine the legends
        left_handles, left_labels = axis_left.get_legend_handles_labels()
        axis_left.legend().remove()
        right_handles, right_labels = axis_right.get_legend_handles_labels()
        axis_right.legend().remove()

        if unique_legend:
            # Determine unique labels and only keep these.
            unique_handles_labels: list[tuple[mlines.Line2D]] = []
            unique_labels: set[str] = set()
            for handle, label in zip(
                left_handles + right_handles, left_labels + right_labels
            ):
                if label not in unique_labels:
                    unique_handles_labels.append((handle, label))
                    unique_labels.add(label)

            plt.legend().remove()
            plt.legend(
                [entry[0] for entry in unique_handles_labels],
                [entry[1] for entry in unique_handles_labels],
                fontsize=7,
            )
        else:
            plt.legend(
                left_handles + right_handles,
                left_labels + right_labels,
                loc="upper right",
                fontsize=7,
            )

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
