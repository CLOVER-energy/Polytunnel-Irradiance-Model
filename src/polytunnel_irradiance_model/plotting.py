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

from .__utils__ import power_spectrum_to_flux_spectrum, power_spectrum_to_par_spectrum
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
    PAR_FLUX = r"PPFD ($\Phi_{\rm{PAR}}$) / $\mu$mol/cm$^2$-nm"
    PAR_IRRADIANCE = "PPFD ($G_{\rm{PAR}}$) / W/m$^{2}$-nm"
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
        spectral_function: callable = power_spectrum_to_flux_spectrum
        label_kwarg: str = "Photon"

    else:
        spectral_function = functools.partial(
            power_spectrum_to_par_spectrum,
            par_wavelength_series=plotting_wavelength_range,
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
            power_spectrum_to_flux_spectrum, wavelength_series=wavelength_range
        )
    elif spectral_units == SpectralUnits.PAR_FLUX:
        spectral_function = functools.partial(
            power_spectrum_to_par_spectrum,
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


#######################
# Plotting code No. 1 #
#######################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# sns.set_context("notebook")

# fig, ax = plt.subplots(figsize=(171*MM, 120*MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     clearsky_total_ground_irradiance_map.iloc[0],
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = max(clearsky_total_ground_irradiance_map.max(axis=0))
# heatmap = sns.heatmap(
#     initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax, cbar_kws={"label": "Irradiance / W/m$^2$"}
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )


# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         clearsky_total_ground_irradiance_map.iloc[time_index], (_dim_x, _dim_y)
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )


# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(clearsky_total_ground_irradiance_map),
#     interval=300,
#     repeat=False,
# )
# ani.save(f"clearsky_total_ground_irradiance_map_{INDEX}.gif", writer="pillow", fps=5)
# plt.show()


#######################
# Plotting code No. 0 #
#######################
#
# try:
#     sns.set_palette(
#         # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
#         [
#             "#423252",
#             "#4A688B",
#             "#779FB1",
#             "#36C7B8",
#             "#FBC412",
#             "#e04606",
#         ],
#     )
# except UnboundLocalError:
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     sns.set_palette(
#         # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
#         [
#             "#423252",
#             "#4A688B",
#             "#779FB1",
#             "#36C7B8",
#             "#FBC412",
#             "#e04606",
#         ],
#     )

# plt.figure(figsize=(171 * MM, 120 * MM))
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_global_spectrum,
#     label=f"Global response ({sum(adjusted_global_spectrum):.4g}×)",
#     color="C5",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_direct_spectrum,
#     label=f"Direct response ({sum(adjusted_direct_spectrum):.4g}×)",
#     color="C4",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_diffuse_spectrum,
#     label=f"Clearsky diffuse response ({sum(adjusted_diffuse_spectrum):.4g}×)",
#     color="C2",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_cloudy_spectrum,
#     label=f"Cloudy-day response ({sum(adjusted_cloudy_spectrum):.4g}×)",
#     color="C0",
# )
# # plt.plot(pyranometer_wavelength_range, _global_spectrum, dashes=(2,4), label="Global solar irradiance", color="C5")
# (right_axis := (left_axis := plt.gca()).twinx()).plot(
#     pyranometer_wavelength_range,
#     pyranometer_response,
#     ":",
#     label="Raw pyranometer response",
#     color="C1",
# )

# left_axis.set_ylabel("Normalised solar spectra")
# right_axis.set_ylabel("Normalised pyranometer response")

# left_axis.tick_params(axis="both", which="major", labelsize=7)
# # right_axis.tick_params(axis="both", which="major", labelsize=7)
# left_axis.set_xlim(0, 4000)

# left_handles, left_labels = left_axis.get_legend_handles_labels()
# right_handles, right_labels = right_axis.get_legend_handles_labels()
# plt.legend(
#     left_handles + right_handles,
#     left_labels + right_labels,
#     loc="upper right",
#     fontsize=7,
# )

# plt.savefig(
#     f"pyranometer_response_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

# plt.figure(figsize=(83 * MM, 60 * MM))
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_global_spectrum,
#     label=f"Global response ({sum(adjusted_global_spectrum):.4g}×)",
#     color="C5",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_direct_spectrum,
#     label=f"Direct response ({sum(adjusted_direct_spectrum):.4g}×)",
#     color="C4",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_diffuse_spectrum,
#     label=f"Clearsky diffuse response ({sum(adjusted_diffuse_spectrum):.4g}×)",
#     color="C2",
# )
# plt.plot(
#     pyranometer_wavelength_range,
#     adjusted_cloudy_spectrum,
#     label=f"Cloudy-day response ({sum(adjusted_cloudy_spectrum):.4g}×)",
#     color="C0",
# )
# # plt.plot(pyranometer_wavelength_range, _global_spectrum, dashes=(2,4), label="Global solar irradiance", color="C5")
# (right_axis := (left_axis := plt.gca()).twinx()).plot(
#     pyranometer_wavelength_range,
#     pyranometer_response,
#     ":",
#     label="Raw pyranometer response",
#     color="C1",
# )

# left_axis.set_ylabel("Normalised solar spectra")
# right_axis.set_ylabel("Normalised pyranometer response")

# left_axis.tick_params(axis="both", which="major", labelsize=7)
# right_axis.tick_params(axis="both", which="major", labelsize=7)
# left_axis.set_xlim(0, 4000)

# left_handles, left_labels = left_axis.get_legend_handles_labels()
# right_handles, right_labels = right_axis.get_legend_handles_labels()
# plt.legend(
#     left_handles + right_handles,
#     left_labels + right_labels,
#     loc="upper right",
#     fontsize=7,
# )

# plt.savefig(
#     f"pyranometer_response_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()


########################
# Plotting code No. 1a #
########################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# sns.set_context("notebook")

# for _index in [40, 50, 60, 70, 80, 90, 100]:
#     fig, ax = plt.subplots(figsize=(171*MM, 120*MM))
#     # Create initial heatmap with dummy data
#     initial_data = np.reshape(
#         clearsky_total_ground_irradiance_map.iloc[_index],
#         (
#             _dim_x := polytunnel.meshgrid_resolution,
#             _dim_y := polytunnel.length_wise_meshgrid_resolution,
#         ),
#     )
#     vmin = 0
#     vmax = max(clearsky_total_ground_irradiance_map.max(axis=0))
#     heatmap = sns.heatmap(
#         initial_data, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=ax, cbar_kws={"label": "Irradiance / W/m$^2$"}
#     )
#     _ten_minutes: int = int(
#         _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
#     )
#     ax.set_title(
#         f"Time index: {_index}. Date: {_index // (_ten_minutes * 24)}; "
#         f"Time: {_index // _ten_minutes}:{int((_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )
#     ax.set_xlabel("Length-wise index")
#     ax.set_ylabel("Width-wise index")
#     plt.savefig(f"clearsky_total_ground_irradiance_map_{_index}_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)

# plt.show()

########################
# Plotting code No. 1b #
########################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# sns.set_context("notebook")

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     clearsky_ground_direct_irradiance_map_sans_pv_module[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = max(
#     direct_day_ground_direct_irradiance.sum(axis=2).max(axis=1)
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         clearsky_ground_direct_irradiance_map_sans_pv_module[time_index].sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(clearsky_ground_direct_irradiance_map_sans_pv_module),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"clearsky_ground_direct_irradiance_map_sans_pv_module_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     clearsky_ground_direct_irradiance_map_with_pv_module[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         clearsky_ground_direct_irradiance_map_with_pv_module[time_index].sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(clearsky_ground_direct_irradiance_map_with_pv_module),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"clearsky_ground_direct_irradiance_map_with_pv_module_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     end_direct_irradiance_map[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         end_direct_irradiance_map[time_index].sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(end_direct_irradiance_map),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"end_direct_irradiance_map_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     direct_day_ground_direct_irradiance[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         direct_day_ground_direct_irradiance[time_index].sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(direct_day_ground_direct_irradiance),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"direct_day_ground_direct_irradiance_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

########################
# Plotting code No. 1c #
########################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# sns.set_context("notebook")

# # Create a slice tool for the PAR wavelengths.
# par_filter: list[bool] = [entry in PAR_WAVELENGTH_RANGE for entry in wavelength_range]

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     clearsky_ground_direct_irradiance_map_sans_pv_module[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = max(
#     (par_filter * direct_day_ground_direct_irradiance).sum(axis=2).max(axis=1)
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         (par_filter * clearsky_ground_direct_irradiance_map_sans_pv_module[time_index]).sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(clearsky_ground_direct_irradiance_map_sans_pv_module),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"clearsky_ground_direct_par_wm2_map_sans_pv_module_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     clearsky_ground_direct_irradiance_map_with_pv_module[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         (par_filter * clearsky_ground_direct_irradiance_map_with_pv_module[time_index]).sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(clearsky_ground_direct_irradiance_map_with_pv_module),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"clearsky_ground_direct_par_wm2_map_with_pv_module_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     end_direct_irradiance_map[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         (par_filter * end_direct_irradiance_map[time_index]).sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(end_direct_irradiance_map),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"end_direct_par_wm2_map_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     direct_day_ground_direct_irradiance[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         (par_filter * direct_day_ground_direct_irradiance[time_index]).sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(direct_day_ground_direct_irradiance),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"direct_day_ground_direct_par_wm2_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

########################
# Plotting code No. 1d #
########################

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import seaborn as sns
# import numpy as np

# sns.set_context("notebook")

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     diffuse_surface_irradiance.iloc[0],
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = direct_day_total_diffuse_surface_irradiance.sum(axis=2).max(axis=1).max()
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         direct_day_total_diffuse_surface_irradiance[time_index].sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:"
#         f"{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(direct_day_total_diffuse_surface_irradiance),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"direct_day_total_diffuse_surface_irradiance_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     diffuse_surface_irradiance.iloc[0],
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = diffuse_day_total_diffuse_surface_irradiance.sum(axis=2).max(axis=1).max()
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         diffuse_day_total_diffuse_surface_irradiance[time_index].sum(axis=1),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:"
#         f"{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(diffuse_day_total_diffuse_surface_irradiance),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"diffuse_day_total_diffuse_surface_irradiance_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     diffuse_day_total_diffuse_surface_irradiance_sans_pv[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         diffuse_day_total_diffuse_surface_irradiance_sans_pv[time_index].sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:"
#         f"{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(diffuse_day_total_diffuse_surface_irradiance_sans_pv),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"diffuse_day_total_diffuse_surface_irradiance_sans_pv_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     direct_day_total_diffuse_surface_irradiance_with_pv[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         direct_day_total_diffuse_surface_irradiance_with_pv[time_index].sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:"
#         f"{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(diffuse_day_total_diffuse_surface_irradiance_sans_pv),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"direct_day_total_diffuse_surface_irradiance__pv_only_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

# fig, ax = plt.subplots(figsize=(171 * MM, 120 * MM))

# # Create initial heatmap with dummy data
# initial_data = np.reshape(
#     direct_day_total_diffuse_surface_irradiance_with_pv[0].sum(axis=1),
#     (
#         _dim_x := polytunnel.meshgrid_resolution,
#         _dim_y := polytunnel.length_wise_meshgrid_resolution,
#     ),
# )
# vmin = 0
# vmax = (
#     direct_day_total_diffuse_surface_irradiance_with_pv.sum(axis=2)
#     .max(axis=1)
#     .max()
# )
# heatmap = sns.heatmap(
#     initial_data,
#     vmin=vmin,
#     vmax=vmax,
#     cmap="viridis",
#     cbar=True,
#     ax=ax,
#     cbar_kws={"label": "Irradiance / W/m$^2$"},
# )

# _ten_minutes: int = int(
#     _ten_minutes := (60 / parsed_args.modelling_temporal_resolution)
# )

# def update(time_index: int):
#     ax.clear()  # clear previous heatmap
#     data = np.reshape(
#         direct_day_total_diffuse_surface_irradiance_with_pv[time_index].sum(
#             axis=1
#         ),
#         (_dim_x, _dim_y),
#     )
#     sns.heatmap(data, vmin=vmin, vmax=vmax, cbar=False, cmap="viridis", ax=ax)
#     ax.set_title(
#         f"Time index: {time_index}. Date: {time_index // (_ten_minutes * 24)}; "
#         f"Time: {time_index // _ten_minutes}:"
#         f"{int((time_index % _ten_minutes) * (6 / _ten_minutes))}0"
#     )

# # Create the animation
# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(diffuse_day_total_diffuse_surface_irradiance_sans_pv),
#     interval=300,
#     repeat=False,
# )
# ani.save(
#     f"direct_day_total_diffuse_surface_irradiance_pv_only_no_vmax_{INDEX}.gif",
#     writer="pillow",
#     fps=5,
# )
# plt.show()

#######################
# Plotting code No. 3 #
#######################

# from matplotlib import colors as m_colors

# # sns.set_palette(
# #     [
# #         "#77AADD",
# #         "#99DDFF",
# #         "#44BB99",
# #         "#BBCC33",
# #         "#AAA000",
# #         "#EEDD88",
# #         "#EE8866",
# #         "#FFAABB",
# #         "#DDDDDD",
# #     ] * ((len(polytunnel.ground_mesh) // 9) + 1)
# # )
# sns.set_palette(sns.cubehelix_palette(start=0.4, rot=-25, n_colors=len(polytunnel.ground_mesh)))
# cmap = sns.color_palette(sns.color_palette().as_hex(), as_cmap=True, n_colors=10)

# plt.figure(figsize=(171 * MM, 120 * MM))
# sns.heatmap(np.reshape(surface_index_of_illuminating_meshpoint.iloc[12], (10, 50)), square=False, cmap=cmap, cbar=False)
# norm = plt.Normalize(
#     0 - 0.5,
#     len(polytunnel.ground_mesh) + 1,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), len(polytunnel.ground_mesh) + 1
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.show()

# #######################
# # Plotting code No. 4 #
# #######################

# from matplotlib import colors as m_colors

# # Plot the spectra at a specific hour for each element within the ground mesh.
# # NOTE: Colours indicate the index of the element providing surface irradiation.

# # Determine the number of non-zero elements
# _hour: int = 12
# num_grid_indices: int = 0
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         num_grid_indices += 1

# sns.set_palette(
#     sns.cubehelix_palette(
#         start=0.4, rot=-1.2, n_colors=num_grid_indices, reverse=True
#     )
# )
# sns.set_palette("viridis", n_colors=num_grid_indices)

# plt.figure(figsize=(171 * MM, 120 * MM))
# dashes = Dashes()
# _zorder = 0
# grid_indices: list[int] = []
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         _color = f"C{grid_index}"
#         _zorder += 1
#         grid_indices.append(grid_index)
#     else:
#         _color = "C0"
#     plt.plot(
#         wavelength_range,
#         clearsky_ground_direct_irradiance_map_with_pv_module_and_spectra[_hour][
#             grid_index
#         ],
#         dashes=next(dashes),
#         label=f"#{_color}" if _color != "C0" else None,
#         color=_color,
#         zorder=0 if _color == "C0" else _zorder,
#     )

# plt.legend().remove()

# norm = plt.Normalize(
#     -0.5,
#     clearsky_ground_direct_irradiance_map_with_pv_module.shape[1] + 0.5,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom",
#         sns.color_palette().as_hex(),
#         num_grid_indices,
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Surface index illuminating",
#     pad=(_pad := 0.125),
# )
# colorbar.set_ticks(grid_indices)
# colorbar.set_ticklabels(
#     [
#         entry if index % 3 == 0 else None
#         for index, entry in enumerate(grid_indices)
#     ]
# )

# axis.tick_params(axis="both", which="major", labelsize=7)
# plt.xlabel("Wavelength / nm", fontsize=7)
# plt.ylabel("Irradiance / W/m$^2$nm", fontsize=7)

# (right_axis := axis.twinx()).plot(
#     wavelength_range,
#     pyranometer_adjusted_interpolated_spectra.direct.values,
#     "--",
#     color="C9",
# )
# right_axis.set_ylabel("Direct-irradiance response / normalised units")
# right_axis.tick_params(axis="both", which="major", labelsize=7)

# plt.savefig(
#     f"ground_through_pv_spectra_profiles_{_hour}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     f"ground_through_pv_spectra_profiles_{_hour}_{INDEX}.png",
#     format="png",
#     bbox_inches="tight",
#     pad_inches=0.05,
#     transparent=True,
#     dpi=1200,
# )
# plt.show()

# # Plot the total irradiance where spectra are coloured based on whether they passed
# # through PV modules (green) or not (yellow) from viridis to match the colours used
# # in the JULIA tmm.

# sns.set_palette(sns.color_palette(["#FDE725", "#21908C", "#31688E", "#440154"]))
# plt.figure(figsize=(171 * MM, 120 * MM))
# _zorder = 0
# _hour = 12
# grid_indices: list[int] = []
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         _color: str = "C1"
#         _dashes: str = "--"
#         _label: str = "Through-PV"
#         grid_indices.append(grid_index)
#     elif sum(clearsky_ground_direct_irradiance_map[_hour][grid_index]) == 0:
#         _color: str = "C3"
#         _dashes = ":"
#         _label = "No direct sunlight"
#     else:
#         _color = "C0"
#         _dashes = ""
#         _label = "Through-polytunnel"
#     plt.plot(
#         wavelength_range,
#         clearsky_ground_direct_irradiance_map[_hour][grid_index],
#         _dashes,
#         label=_label,
#         color=_color,
#         zorder=0 if _color == "C0" else 1,
#         alpha=0.3,
#     )

# # Determine unique labels and only keep these.
# handles, labels = (axis := plt.gca()).get_legend_handles_labels()
# unique_handles_labels: list[tuple[mlines.Line2D]] = []
# unique_labels: set[str] = set()
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_handles_labels.append((handle, label))
#         unique_labels.add(label)

# plt.legend().remove()
# plt.legend(
#     [entry[0] for entry in unique_handles_labels],
#     [entry[1] for entry in unique_handles_labels],
#     fontsize=7,
# )

# axis.tick_params(axis="both", which="major", labelsize=7)
# plt.xlabel("Wavelength / nm", fontsize=7)
# plt.ylabel("Irradiance / W/m$^2$nm", fontsize=7)

# (right_axis := axis.twinx()).plot(
#     wavelength_range,
#     pyranometer_adjusted_interpolated_spectra.direct.values,
#     "--",
#     color="C9",
# )
# right_axis.set_ylabel("Direct-irradiance response / normalised units")
# right_axis.tick_params(axis="both", which="major", labelsize=7)

# plt.savefig(
#     f"ground_total_with_pv_spectra_profiles_{_hour}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     f"ground_total_with_pv_spectra_profiles_{_hour}_{INDEX}.png",
#     format="png",
#     bbox_inches="tight",
#     pad_inches=0.05,
#     transparent=True,
#     dpi=1200,
# )
# plt.show()

# sns.set_palette(sns.color_palette(["#FDE725", "#21908C", "#31688E", "#440154"]))
# plt.figure(figsize=(171 * MM, 120 * MM))
# _zorder = 0
# _hour = 12
# grid_indices: list[int] = []
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         _color: str = "C1"
#         _dashes: str = "--"
#         _label: str = "Through-PV"
#         grid_indices.append(grid_index)
#     elif sum(clearsky_ground_direct_irradiance_map[_hour][grid_index]) == 0:
#         _color: str = "C3"
#         _dashes = ":"
#         _label = "No direct sunlight"
#     else:
#         _color = "C0"
#         _dashes = ""
#         _label = "Through-polytunnel"
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(
#             clearsky_ground_direct_irradiance_map[_hour][grid_index],
#             wavelength_range,
#         ),
#         _dashes,
#         label=_label,
#         color=_color,
#         zorder=0 if _color == "C0" else 1,
#         alpha=0.3,
#     )

# # Determine unique labels and only keep these.
# handles, labels = (axis := plt.gca()).get_legend_handles_labels()
# unique_handles_labels: list[tuple[mlines.Line2D]] = []
# unique_labels: set[str] = set()
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_handles_labels.append((handle, label))
#         unique_labels.add(label)

# axis.tick_params(axis="both", which="major", labelsize=7)
# plt.xlabel("Wavelength / nm", fontsize=7)
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})

# # (right_axis := axis.twinx()).plot(
# #     wavelength_range,        sns.set_palette(sns.color_palette(["#FDE725", "#21908C", "#31688E", "#440154"]))
# plt.figure(figsize=(171 * MM, 120 * MM))
# _zorder = 0
# _hour = 12
# grid_indices: list[int] = []
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         _color: str = "C1"
#         _dashes: str = "--"
#         _label: str = "Through-PV"
#         grid_indices.append(grid_index)
#     elif sum(clearsky_ground_direct_irradiance_map[_hour][grid_index]) == 0:
#         _color: str = "C3"
#         _dashes = ":"
#         _label = "No direct sunlight"
#     else:
#         _color = "C0"
#         _dashes = ""
#         _label = "Through-polytunnel"
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(
#             clearsky_ground_direct_irradiance_map[_hour][grid_index],
#             wavelength_range,
#         ),
#         _dashes,
#         label=_label,
#         color=_color,
#         zorder=0 if _color == "C0" else 1,
#         alpha=0.3,
#     )

# # Determine unique labels and only keep these.
# handles, labels = (axis := plt.gca()).get_legend_handles_labels()
# unique_handles_labels: list[tuple[mlines.Line2D]] = []
# unique_labels: set[str] = set()
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_handles_labels.append((handle, label))
#         unique_labels.add(label)

# axis.tick_params(axis="both", which="major", labelsize=7)
# plt.xlabel("Wavelength / nm", fontsize=7)
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})

# # (right_axis := axis.twinx()).plot(
# #     wavelength_range,
# #     spectrum_to_flux(pyranometer_adjusted_interpolated_spectra.direct.values, wavelength_range),
# #     "--",
# #     color="C2",
# #     label="Incident spectrum"
# # )
# # right_axis.set_ylabel("Direct-irradiance response / normalised units")
# # right_axis.tick_params(axis="both", which="major", labelsize=7)
# # right_labels, right_handles = right_axis.get_legend_handles_labels()

# plt.legend().remove()
# plt.legend(
#     [entry[0] for entry in unique_handles_labels], # + right_labels,
#     [entry[1] for entry in unique_handles_labels], # + right_handles,
#     fontsize=7,
# )

# plt.savefig(
#     f"ground_total_with_pv_flux_profiles_{_hour}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     f"ground_total_with_pv_flux_profiles_{_hour}_{INDEX}.png",
#     format="png",
#     bbox_inches="tight",
#     pad_inches=0.05,
#     transparent=True,
#     dpi=1200,
# )
# plt.show()

# plt.figure(figsize=(73 * MM, 60 * MM))
# _zorder = 0
# _hour = 12
# grid_indices: list[int] = []
# for grid_index in range(len(clearsky_ground_direct_irradiance_map[_hour])):
#     if (
#         clearsky_ground_direct_irradiance_map_with_pv_module[grid_index][_hour]
#         > 0
#     ):
#         _color: str = "C1"
#         _dashes: str = "--"
#         _label: str = "Through-PV"
#         grid_indices.append(grid_index)
#     elif sum(clearsky_ground_direct_irradiance_map[_hour][grid_index]) == 0:
#         _color: str = "C3"
#         _dashes = ":"
#         _label = "No direct sunlight"
#     else:
#         _color = "C0"
#         _dashes = ""
#         _label = "Through-polytunnel"
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(
#             clearsky_ground_direct_irradiance_map[_hour][grid_index],
#             wavelength_range,
#         ),
#         _dashes,
#         label=_label,
#         color=_color,
#         zorder=0 if _color == "C0" else 1,
#         alpha=0.3,
#     )

# # Determine unique labels and only keep these.
# handles, labels = (axis := plt.gca()).get_legend_handles_labels()
# unique_handles_labels: list[tuple[mlines.Line2D]] = []
# unique_labels: set[str] = set()
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_handles_labels.append((handle, label))
#         unique_labels.add(label)

# axis.tick_params(axis="both", which="major", labelsize=7)
# plt.xlabel("Wavelength / nm", fontsize=7)
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})

# # (right_axis := axis.twinx()).plot(
# #     wavelength_range,
# #     spectrum_to_flux(pyranometer_adjusted_interpolated_spectra.direct.values, wavelength_range),
# #     "--",
# #     color="C2",
# #     label="Incident spectrum"
# # )
# # right_axis.set_ylabel("Direct-irradiance response / normalised units")
# # right_axis.tick_params(axis="both", which="major", labelsize=7)
# # right_labels, right_handles = right_axis.get_legend_handles_labels()

# plt.legend().remove()
# plt.legend(
#     [entry[0] for entry in unique_handles_labels], # + right_labels,
#     [entry[1] for entry in unique_handles_labels], # + right_handles,
#     fontsize=7,
# )

# plt.savefig(
#     f"ground_total_with_pv_flux_profiles_small_{_hour}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     f"ground_total_with_pv_flux_profiles_small_{_hour}_{INDEX}.png",
#     format="png",
#     bbox_inches="tight",
#     pad_inches=0.05,
#     transparent=True,
#     dpi=1200,
# )
# plt.show()

#######################
# Plotting code No. 5 #
#######################

# # Plot an average power spectrum over the ground.
# hourly_spectra = {
#     index: SpectralDistribution(
#         (spectrum := direct_day_ground_direct_irradiance.mean(axis=1)[index])
#         / max(spectrum),
#         wavelength_range,
#         name=f"{index}:00",
#     )
#     for index in range(len(direct_day_ground_direct_irradiance.mean(axis=1)))
#     if direct_day_ground_direct_irradiance[index].mean(axis=1).sum() > 0
# }
# # for index, spectrum in hourly_spectra.items():
# #     colour.plotting.plot_single_sd(spectrum)

# sns.set_palette(
#     [
#         "#77AADD",
#         "#99DDFF",
#         "#44BB99",
#         "#BBCC33",
#         "#AAA000",
#         "#EEDD88",
#         "#EE8866",
#         "#FFAABB",
#         "#DDDDDD",
#     ]
# )

# fig, axis = colour.plotting.plot_multi_sds(
#     list(hourly_spectra.values()),
#     figsize=(171 * MM, 120 * MM),
#     plot_kwargs=[{"alpha": 0.8}] * len(sns.color_palette())
#     + [{"use_sd_colours": False, "linestyle": "dashed", "alpha": 0.8}]
#     * (len(hourly_spectra) - len(sns.color_palette())),
# )
# axis.tick_params(axis="both", which="major", labelsize=7)
# axis.set_xlabel(axis.get_xlabel(), fontsize=7)
# axis.set_ylabel(axis.get_ylabel(), fontsize=7)
# axis.legend(*axis.get_legend_handles_labels(), fontsize=7, title="Time of day", title_fontsize=7)
# fig.savefig(f"average_direct_ground_power_spectrum_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)

# hourly_spectra = {
#     index: SpectralDistribution(
#         spectrum_to_flux(direct_day_ground_direct_irradiance.mean(axis=1)[index], wavelength_range),
#         wavelength_range,
#         name=f"{index}:00",
#     )
#     for index in range(len(direct_day_ground_direct_irradiance.mean(axis=1)))
#     if direct_day_ground_direct_irradiance[index].mean(axis=1).sum() > 0
# }
# # for index, spectrum in hourly_spectra.items():
# #     colour.plotting.plot_single_sd(spectrum)

# sns.set_palette(
#     [
#         "#77AADD",
#         "#99DDFF",
#         "#44BB99",
#         "#BBCC33",
#         "#AAA000",
#         "#EEDD88",
#         "#EE8866",
#         "#FFAABB",
#         "#DDDDDD",
#     ]
# )

# fig, axis = colour.plotting.plot_multi_sds(
#     list(hourly_spectra.values()),
#     figsize=(171 * MM, 120 * MM),
#     plot_kwargs=[{"alpha": 0.8}] * len(sns.color_palette())
#     + [{"use_sd_colours": False, "linestyle": "dashed", "alpha": 0.8}]
#     * (len(hourly_spectra) - len(sns.color_palette())),
# )
# axis.tick_params(axis="both", which="major", labelsize=7)
# axis.set_xlabel(axis.get_xlabel(), fontsize=7)
# axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontsize=7)
# axis.legend(*axis.get_legend_handles_labels(), fontsize=7, title="Time of day", title_fontsize=7)
# fig.savefig(f"average_direct_ground_flux_spectrum_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)

# fig, axis = colour.plotting.plot_multi_sds(
#     list(hourly_spectra.values()),
#     figsize=(83 * MM, 60 * MM),
#     plot_kwargs=[{"alpha": 0.8}] * len(sns.color_palette())
#     + [{"use_sd_colours": False, "linestyle": "dashed", "alpha": 0.8}]
#     * (len(hourly_spectra) - len(sns.color_palette())),
# )
# axis.tick_params(axis="both", which="major", labelsize=7)
# axis.set_xlabel(axis.get_xlabel(), fontsize=7)
# axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontsize=7)
# axis.legend(*axis.get_legend_handles_labels(), fontsize=7, ncols=2, title="Time of day", title_fontsize=7)
# fig.set_size_inches((83 * MM, 60 * MM))
# fig.savefig(f"average_direct_ground_flux_spectrum_small_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)

# colour.plotting.plot_multi_sds(
#     [
#         SpectralDistribution(
#             pyranometer_adjusted_interpolated_spectra[column], wavelength_range
#         )
#         for column in pyranometer_adjusted_interpolated_spectra
#     ]
# )

#######################
# Plotting code No. 6 #
#######################

# fig, axes = plt.subplots(5, 1, figsize=(83 * MM, 120 * MM))
# fig.subplots_adjust(hspace=0.25)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[0], (10, 50)),
#     cmap="viridis",
#     ax=axes[0],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[165], (10, 50)),
#     cmap="viridis",
#     ax=axes[1],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[250], (10, 50)),
#     cmap="viridis",
#     ax=axes[2],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     cbar_kws={"label": "Projection factor"},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[275], (10, 50)),
#     cmap="viridis",
#     ax=axes[3],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[410], (10, 50)),
#     cmap="viridis",
#     ax=axes[4],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# plt.xlabel("Depth index", fontsize=7)
# axes[2].set_ylabel("Width index", fontsize=7)
# axes[0].set_title("Index 0", fontsize=7, fontweight="bold")
# axes[1].set_title("Index 165", fontsize=7, fontweight="bold")
# axes[2].set_title("Index 250", fontsize=7, fontweight="bold")
# axes[3].set_title("Index 275", fontsize=7, fontweight="bold")
# axes[4].set_title("Index 410", fontsize=7, fontweight="bold")
# for axis in axes[:-1]:
#     axis.tick_params(bottom=False, labelbottom=False)

# for axis in axes:
#     axis.tick_params(which="both", size=7, labelsize=7)
#     sns.despine(fig, axis, bottom=True)
#     # axis.set_ylabel("Width index", fontsize=7)

# for label, axis in zip(["a", "b", "c", "d", "e"], axes):
#     axis.text(
#         -0.12,
#         1.35,
#         f"{label}.",
#         transform=axis.transAxes,
#         fontsize=7,
#         fontweight="bold",
#         va="top",
#         ha="right",
#     )

# plt.savefig(
#     f"distance_example_heatmap_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# fig, axes = plt.subplots(5, 1, figsize=(171 * MM, 171 * MM))
# fig.subplots_adjust(hspace=0.25)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[0], (10, 50)),
#     cmap="viridis",
#     ax=axes[0],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[165], (10, 50)),
#     cmap="viridis",
#     ax=axes[1],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[250], (10, 50)),
#     cmap="viridis",
#     ax=axes[2],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     cbar_kws={"label": "Projection factor"},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[275], (10, 50)),
#     cmap="viridis",
#     ax=axes[3],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[410], (10, 50)),
#     cmap="viridis",
#     ax=axes[4],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# plt.xlabel("Depth index", fontsize=7)
# axes[2].set_ylabel("Width index", fontsize=7)
# axes[0].set_title("Index 0", fontsize=7, fontweight="bold")
# axes[1].set_title("Index 165", fontsize=7, fontweight="bold")
# axes[2].set_title("Index 250", fontsize=7, fontweight="bold")
# axes[3].set_title("Index 275", fontsize=7, fontweight="bold")
# axes[4].set_title("Index 410", fontsize=7, fontweight="bold")
# for axis in axes[:-1]:
#     axis.tick_params(bottom=False, labelbottom=False)

# for axis in axes:
#     axis.tick_params(which="both", size=7, labelsize=7)
#     sns.despine(fig, axis, bottom=True)
#     # axis.set_ylabel("Width index", fontsize=7)

# for label, axis in zip(["a", "b", "c", "d", "e"], axes):
#     axis.text(
#         -0.08,
#         1.1,
#         f"{label}.",
#         transform=axis.transAxes,
#         fontsize=7,
#         fontweight="bold",
#         va="top",
#         ha="right",
#     )

# plt.savefig(
#     f"distance_example_heatmap_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# fig, axes = plt.subplots(5, 1, figsize=(83 * MM, 120 * MM))
# fig.subplots_adjust(hspace=0.25)
# _norm = m_colors.LogNorm(0.00001, 0.0325)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[0], (10, 50)),
#     cmap="viridis",
#     ax=axes[0],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[165], (10, 50)),
#     cmap="viridis",
#     ax=axes[1],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[250], (10, 50)),
#     cmap="viridis",
#     ax=axes[2],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     cbar_kws={"label": "Projection factor"},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[275], (10, 50)),
#     cmap="viridis",
#     ax=axes[3],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[410], (10, 50)),
#     cmap="viridis",
#     ax=axes[4],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# plt.xlabel("Depth index", fontsize=7)
# axes[2].set_ylabel("Width index", fontsize=7)
# axes[0].set_title("Index 0", fontsize=7, fontweight="bold")
# axes[1].set_title("Index 165", fontsize=7, fontweight="bold")
# axes[2].set_title("Index 250", fontsize=7, fontweight="bold")
# axes[3].set_title("Index 275", fontsize=7, fontweight="bold")
# axes[4].set_title("Index 410", fontsize=7, fontweight="bold")
# for axis in axes[:-1]:
#     axis.tick_params(bottom=False, labelbottom=False)

# for axis in axes:
#     axis.tick_params(which="both", size=7, labelsize=7)
#     sns.despine(fig, axis, bottom=True)
#     # axis.set_ylabel("Width index", fontsize=7)

# for label, axis in zip(["a", "b", "c", "d", "e"], axes):
#     axis.text(
#         -0.12,
#         1.35,
#         f"{label}.",
#         transform=axis.transAxes,
#         fontsize=7,
#         fontweight="bold",
#         va="top",
#         ha="right",
#     )

# plt.savefig(
#     f"distance_example_heatmap_small_log_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# fig, axes = plt.subplots(5, 1, figsize=(171 * MM, 171 * MM))
# fig.subplots_adjust(hspace=0.25)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[0], (10, 50)),
#     cmap="viridis",
#     ax=axes[0],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[165], (10, 50)),
#     cmap="viridis",
#     ax=axes[1],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[250], (10, 50)),
#     cmap="viridis",
#     ax=axes[2],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
#     cbar_kws={"label": "Projection factor"},
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[275], (10, 50)),
#     cmap="viridis",
#     ax=axes[3],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# heatmap = sns.heatmap(
#     np.reshape(ground_to_surface_projection_frame[410], (10, 50)),
#     cmap="viridis",
#     ax=axes[4],
#     square=True,
#     vmax=0.0325,
#     annot_kws={"fontsize": 7},
#     norm=_norm,
# )
# heatmap.figure.axes[-1].tick_params(which="both", labelsize=7)
# plt.xlabel("Depth index", fontsize=7)
# axes[2].set_ylabel("Width index", fontsize=7)
# axes[0].set_title("Index 0", fontsize=7, fontweight="bold")
# axes[1].set_title("Index 165", fontsize=7, fontweight="bold")
# axes[2].set_title("Index 250", fontsize=7, fontweight="bold")
# axes[3].set_title("Index 275", fontsize=7, fontweight="bold")
# axes[4].set_title("Index 410", fontsize=7, fontweight="bold")
# for axis in axes[:-1]:
#     axis.tick_params(bottom=False, labelbottom=False)

# for axis in axes:
#     axis.tick_params(which="both", size=7, labelsize=7)
#     sns.despine(fig, axis, bottom=True)
#     # axis.set_ylabel("Width index", fontsize=7)

# for label, axis in zip(["a", "b", "c", "d", "e"], axes):
#     axis.text(
#         -0.08,
#         1.1,
#         f"{label}.",
#         transform=axis.transAxes,
#         fontsize=7,
#         fontweight="bold",
#         va="top",
#         ha="right",
#     )

# plt.savefig(
#     f"distance_example_heatmap_log_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

#######################
# Plotting code No. 7 #
#######################

# plt.figure(figsize=(171 * MM, 120 * MM))
# sns.set_palette(sns.blend_palette(["#36C7B8", "#423252"], n_colors=50))
# _hour: int = 12

# for entry in direct_day_total_diffuse_surface_irradiance[_hour]:
#     plt.plot(
#         wavelength_range, spectrum_to_flux(entry, wavelength_range), alpha=0.3
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Length-wise surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.legend().remove()
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.savefig(
#     f"surface_direct_day_diffuse_spectra_z_wise_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(171 * MM, 120 * MM))
# sns.set_palette(sns.blend_palette(["#36C7B8", "#423252"], n_colors=10))
# _hour: int = 12

# for index, entry in enumerate(
#     direct_day_total_diffuse_surface_irradiance[_hour]
# ):
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(entry, wavelength_range),
#         alpha=0.3,
#         color=f"C{index // 50}",
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Rotational surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.legend().remove()
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.savefig(
#     f"surface_direct_day_diffuse_spectra_phi_wise_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.set_palette(sns.blend_palette(["#36C7B8", "#423252"], n_colors=50))
# _hour: int = 12

# for entry in direct_day_total_diffuse_surface_irradiance[_hour]:
#     plt.plot(
#         wavelength_range, spectrum_to_flux(entry, wavelength_range), alpha=0.3
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Length-wise surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.legend().remove()
# plt.savefig(
#     f"surface_direct_day_diffuse_spectra_z_wise_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.set_palette(sns.blend_palette(["#36C7B8", "#423252"], n_colors=10))
# _hour: int = 12

# for index, entry in enumerate(
#     direct_day_total_diffuse_surface_irradiance[_hour]
# ):
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(entry, wavelength_range),
#         alpha=0.3,
#         color=f"C{index // 50}",
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Rotational surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.legend().remove()
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.savefig(
#     f"surface_direct_day_diffuse_spectra_phi_wise_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(171 * MM, 120 * MM))
# sns.set_palette(sns.blend_palette(["#dbe9f6", "#4A688B"], n_colors=50))
# _hour: int = 12

# for entry in diffuse_day_total_diffuse_surface_irradiance[_hour]:
#     plt.plot(
#         wavelength_range, spectrum_to_flux(entry, wavelength_range), alpha=0.3
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Length-wise surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.legend().remove()
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.savefig(
#     f"surface_diffuse_day_diffuse_spectra_z_wise_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(171 * MM, 120 * MM))
# sns.set_palette(sns.blend_palette(["#dbe9f6", "#4A688B"], n_colors=10))
# _hour: int = 12

# for index, entry in enumerate(
#     diffuse_day_total_diffuse_surface_irradiance[_hour]
# ):
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(entry, wavelength_range),
#         alpha=0.3,
#         color=f"C{index // 50}",
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Rotational surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.legend().remove()
# plt.savefig(
#     f"surface_diffuse_day_diffuse_spectra_phi_wise_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.set_palette(sns.blend_palette(["#dbe9f6", "#4A688B"], n_colors=50))
# _hour: int = 12

# for entry in diffuse_day_total_diffuse_surface_irradiance[_hour]:
#     plt.plot(
#         wavelength_range, spectrum_to_flux(entry, wavelength_range), alpha=0.3
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Length-wise surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.legend().remove()
# plt.savefig(
#     f"surface_diffuse_day_diffuse_spectra_z_wise_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.set_palette(sns.blend_palette(["#dbe9f6", "#4A688B"], n_colors=10))
# _hour: int = 12

# for index, entry in enumerate(
#     diffuse_day_total_diffuse_surface_irradiance[_hour]
# ):
#     plt.plot(
#         wavelength_range,
#         spectrum_to_flux(entry, wavelength_range),
#         alpha=0.3,
#         color=f"C{index // 50}",
#     )

# norm = plt.Normalize(
#     -0.5,
#     50.51,
# )
# scalar_mappable = plt.cm.ScalarMappable(
#     cmap=m_colors.LinearSegmentedColormap.from_list(
#         "Custom", sns.color_palette().as_hex(), 50
#     ),
#     norm=norm,
# )

# colorbar = (axis := plt.gca()).figure.colorbar(
#     scalar_mappable,
#     ax=axis,
#     label="Rotational surface-mesh index",
#     pad=(_pad := 0.025),
# )
# plt.xlabel("Wavelength ($\lambda$) / nm")
# plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
# plt.legend().remove()
# plt.savefig(
#     f"surface_diffuse_day_diffuse_spectra_phi_wise_small_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.show()

#######################
# Plotting code No. 6 #
#######################

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.boxplot(
#     dif_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
#     boxprops=dict(alpha=0.75),
#     color="C3",
#     label="Diffuse-day prediction",
#     saturation=1,
#     # linecolor="C3",
#     zorder=0,
# )
# sns.boxplot(
#     dir_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
#     boxprops=dict(alpha=0.75),
#     color="C4",
#     label="Direct-day prediction",
#     # linecolor="C4",
#     saturation=1,
#     zorder=0,
# )
# sns.scatterplot(
#     x=range(len(dir_day_gnd_tot_val)),
#     y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     color="C0",
#     label="Total PAR",
#     marker="H",
#     s=60,
#     zorder=1,
# )
# plt.plot(
#     range(len(dir_day_gnd_tot_val)),
#     dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     color="C0",
#     zorder=1,
# )
# plt.errorbar(
#     x=range(len(dir_day_gnd_tot_val)),
#     y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value]
#     * 0.48,
#     ls="none",
#     color="C0",
#     zorder=1,
# )
# plt.xlabel("Time / h")
# plt.ylabel(r"PPFD ($\Phi$) / $\mu$mol/m$^2$")

# axis_right = (axis_left := plt.gca()).twinx()
# axis_left.tick_params(axis="both", which="major", labelsize=7)
# axis_right.tick_params(axis="both", which="major", labelsize=7)
# axis_right.set_ylabel("Diffusivity")
# sns.scatterplot(
#     x=range(len(dir_day_gnd_tot_val)),
#     y=dir_day_gnd_tot_val["diffusivity"],
#     alpha=0.7,
#     color="C1",
#     label="Diffusivity",
#     marker="D",
#     s=40,
#     zorder=1,
# )
# left_handles, left_labels = axis_left.get_legend_handles_labels()
# axis_left.legend().remove()
# right_handles, right_labels = axis_right.get_legend_handles_labels()
# axis_right.legend().remove()

# plt.legend(
#     left_handles + right_handles,
#     left_labels + right_labels,
#     loc="upper right",
#     fontsize=7,
# )
# axis_right.set_ylim(-0.05, 1.05)
# axis_left.set_ylim(-25, 1600)

# plt.xticks(
#     plt.xticks()[0][::3],
#     [entry for entry in dir_day_gnd_tot_val.index][::3],
# )

# plt.savefig(
#     "validation_total_map_boxplot_"
#     f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
#     f"{parsed_args.start_time.replace(':','_')}_"
#     f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# pbar.update(1)

# #######################
# # Plotting code No. 7 #
# #######################

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.boxplot(
#     dif_day_gnd_tot_val.reset_index(drop=True).transpose()[:-13],
#     boxprops=dict(alpha=0.75),
#     color="C3",
#     label="Diffuse-day prediction",
#     saturation=1,
#     # linecolor="C3",
#     zorder=0,
# )
# sns.boxplot(
#     dir_day_gnd_dif_val.reset_index(drop=True).transpose()[:-13],
#     boxprops=dict(alpha=0.75),
#     color="C4",
#     label="Direct-day prediction",
#     # linecolor="C4",
#     saturation=1,
#     zorder=0,
# )
# sns.scatterplot(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
#     color="C1",
#     label="Diffuse PAR",
#     marker="H",
#     s=60,
#     zorder=1,
# )
# plt.plot(
#     range(len(dir_day_gnd_dir_val)),
#     dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
#     color="C1",
#     zorder=1,
# )
# plt.errorbar(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_PAR.value] * 0.48,
#     yerr=dir_day_gnd_dir_val[ValidationColumns.DIFFUSE_ERROR.value]
#     * 0.48,
#     ls="none",
#     color="C1",
#     zorder=1,
# )
# plt.xlabel("Time / h")
# plt.ylabel(r"PPFD ($\Phi$) / $\mu$mol/m$^2$")

# axis_right = (axis_left := plt.gca()).twinx()
# axis_left.tick_params(axis="both", which="major", labelsize=7)
# axis_right.tick_params(axis="both", which="major", labelsize=7)
# sns.scatterplot(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val["diffusivity"],
#     alpha=0.7,
#     color="C1",
#     label="Diffusivity",
#     marker="D",
#     s=40,
#     zorder=1,
# )
# left_handles, left_labels = axis_left.get_legend_handles_labels()
# axis_left.legend().remove()
# right_handles, right_labels = axis_right.get_legend_handles_labels()
# axis_right.legend().remove()
# axis_right.set_ylabel("Diffusivity")

# plt.legend(
#     left_handles + right_handles,
#     left_labels + right_labels,
#     loc="upper right",
#     fontsize=7,
# )
# axis_right.set_ylim(-0.05, 1.05)
# axis_left.set_ylim(-25, 1600)

# plt.xticks(
#     plt.xticks()[0][::3],
#     [entry for entry in dir_day_gnd_dir_val.index][::3],
# )

# plt.savefig(
#     "validation_diffuse_map_boxplot_"
#     f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
#     f"{parsed_args.start_time.replace(':','_')}_"
#     f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# pbar.update(1)

# #######################
# # Plotting code No. 8 #
# #######################

# plt.figure(figsize=(83 * MM, 60 * MM))
# sns.scatterplot(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val.reset_index(drop=True)
#     .transpose()[:-13]
#     .mean(axis=0),
#     # boxprops=dict(alpha=0.75),
#     color="C4",
#     label="Direct-day prediction",
#     # linecolor="C4",
#     marker="h",
#     s=60,
#     # saturation=1,
#     zorder=0,
# )
# plt.plot(
#     range(len(dir_day_gnd_dir_val)),
#     dir_day_gnd_dir_val.reset_index(drop=True)
#     .transpose()[:-13]
#     .mean(axis=0),
#     color="C4",
#     zorder=0,
# )
# sns.scatterplot(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
#     color="C1",
#     label="Direct PAR",
#     marker="H",
#     s=60,
#     zorder=1,
# )
# plt.plot(
#     range(len(dir_day_gnd_dir_val)),
#     dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
#     color="C1",
#     zorder=1,
# )
# plt.errorbar(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val[ValidationColumns.DIRECT_PAR.value] * 0.48,
#     yerr=dir_day_gnd_dir_val[ValidationColumns.DIRECT_ERROR.value]
#     * 0.48,
#     ls="none",
#     color="C1",
#     zorder=1,
# )
# plt.xlabel("Time / h")
# plt.ylabel(r"PPFD ($\Phi$) / $\mu$mol/m$^2$")

# axis_right = (axis_left := plt.gca()).twinx()
# axis_left.tick_params(axis="both", which="major", labelsize=7)
# axis_right.tick_params(axis="both", which="major", labelsize=7)
# axis_right.set_ylabel("Diffusivity")
# sns.scatterplot(
#     x=range(len(dir_day_gnd_dir_val)),
#     y=dir_day_gnd_dir_val["diffusivity"],
#     alpha=0.7,
#     color="C1",
#     label="Diffusivity",
#     marker="D",
#     s=40,
#     zorder=1,
# )
# left_handles, left_labels = axis_left.get_legend_handles_labels()
# axis_left.legend().remove()
# right_handles, right_labels = axis_right.get_legend_handles_labels()
# axis_right.legend().remove()

# plt.legend(
#     left_handles + right_handles,
#     left_labels + right_labels,
#     loc="upper right",
#     fontsize=7,
# )
# axis_right.set_ylim(-0.05, 1.05)
# axis_left.set_ylim(-25, 1600)

# plt.xticks(
#     list(range(len(dir_day_gnd_dir_val.index)))[::3],
#     [entry for entry in dir_day_gnd_dir_val.index][::3],
# )

# plt.savefig(
#     "validation_direct_map_boxplot_"
#     f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
#     f"{parsed_args.start_time.replace(':','_')}_"
#     f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# pbar.update(1)

# #######################
# # Plotting code No. 9 #
# #######################

# # Compute the cloudiness based on the on-the-ground PAR seen.
# diffusivity: pd.Series = (
#     dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48
#     - dir_day_gnd_tot_val[parsed_args.validation_index]
# ) / (
#     dif_day_gnd_tot_val[parsed_args.validation_index]
#     - dir_day_gnd_tot_val[parsed_args.validation_index]
# )

# diffusivity_error = abs(diffusivity * 0.1)

# plt.figure(figsize=(83 * MM, 60 * MM))
# axis_right = (axis_left := plt.gca()).twinx()

# sns.scatterplot(
#     x=range(len(dir_day_gnd_tot_val)),
#     y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     ax=axis_left,
#     color="C0",
#     label="Total PAR",
#     marker="H",
#     s=60,
#     zorder=1,
# )
# axis_left.plot(
#     range(len(dir_day_gnd_tot_val)),
#     dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     color="C0",
#     zorder=1,
# )
# axis_left.errorbar(
#     x=range(len(dir_day_gnd_tot_val)),
#     y=dir_day_gnd_tot_val[ValidationColumns.TOTAL_PAR.value] * 0.48,
#     yerr=dir_day_gnd_dir_val[ValidationColumns.TOTAL_ERROR.value]
#     * 0.48,
#     ls="none",
#     color="C0",
#     zorder=1,
# )
# plt.xlabel("Time / h")
# plt.ylabel(r"PPFD ($\Phi$) / $\mu$mol/m$^2$")

# sns.scatterplot(
#     x=range(len(diffusivity)),
#     y=diffusivity,
#     alpha=0.7,
#     ax=axis_right,
#     color="C2",
#     label="Predicted weather diffusivity",
#     marker="X",
#     s=40,
#     zorder=1,
# )
# axis_right.errorbar(
#     x=(x_range := range(len(dir_day_gnd_dir_val))),
#     y=diffusivity,
#     yerr=diffusivity_error,
#     ls="none",
#     color="C2",
#     zorder=1,
# )
# axis_right.set_xlabel("Time / h")
# axis_left.set_ylabel(r"PPFD ($\Phi$) / $\mu$mol/m$^2$")
# axis_right.set_ylabel("Diffusivity")

# plt.xticks(
#     list(range(len(dir_day_gnd_dir_val.index)))[::4],
#     [entry for entry in dir_day_gnd_dir_val.index][::4],
# )

# lower_ylim: float = -0.75
# upper_ylim: float = 4.75
# axis_right.fill_between(
#     x_range,
#     [lower_ylim] * len(x_range),
#     [0] * len(x_range),
#     alpha=0.3,
#     color="grey",
#     hatch="//",
#     zorder=0,
#     label="Out-of-bounds result",
# )
# axis_right.fill_between(
#     x_range,
#     [1] * len(x_range),
#     [upper_ylim] * len(x_range),
#     alpha=0.3,
#     color="grey",
#     hatch="//",
#     zorder=0,
# )

# axis_left.set_ylim(-25, 1600)
# axis_right.set_ylim(lower_ylim, upper_ylim)

# handles_l, labels_l = axis_left.get_legend_handles_labels()
# handles_r, labels_r = axis_right.get_legend_handles_labels()

# axis_left.tick_params(axis="both", which="major", labelsize=7)
# axis_right.tick_params(axis="both", which="major", labelsize=7)

# axis_left.legend().remove()
# axis_right.legend().remove()
# axis_right.legend(handles_l + handles_r, labels_l + labels_r)

# plt.savefig(
#     "validation_diffusivity_prediction_"
#     f"{polytunnel_diffusivity}_{polytunnel.name}_{alt_weather}"
#     f"{parsed_args.start_time.replace(':','_')}_"
#     f"{parsed_args.end_time.replace(':','_')}_{INDEX}.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# pbar.update(1)
