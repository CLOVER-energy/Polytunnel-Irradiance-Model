########################################################################################
# tmm_ppv.py --- TMM code in Python: will either utilise julia base or tmm Python code #
#                                                                                      #
# Author(s): Benedict Winchester                                                       #
# Date created: Spring 2026                                                            #
#                                                                                      #
########################################################################################

import json
import os

from dataclasses import dataclass
from matplotlib import rc, rcParams

import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
from tqdm import tqdm
from tmm import (
    absorp_in_each_layer,
    coh_tmm,
    ellips,
    find_in_structure_with_inf,
    inc_tmm,
    position_resolved,
    unpolarized_RT,
)

# Plotting context
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7

sns.set_palette(
    bright_palette := [
        "#3455CC",
        "#557BF9",
        "#9FB0FC",
        "#BFCAFD",
        "#FFF1D4",
        "#FBC412",
        "#FE8224",
        "#E03944",
    ]
)

# ACTIVE_LAYER:
#   Label used for the active layer.
ACTIVE_LAYER: str = "Active layer"

# BACKGROUND_PALETTE:
#   The palette to use for the background absorption for the layers.
BACKGROUND_PALETTE = [
    tuple(sub_entry / 255 for sub_entry in entry)
    for entry in [
        (221, 221, 221),
        (46, 37, 133),
        (51, 117, 56),
        (93, 168, 153),
        (148, 203, 236),
        (220, 205, 125),
        (194, 106, 119),
        (159, 74, 150),
        (126, 41, 84),
    ]
]


# INDEX:
#   Variable for labelling plots.
INDEX: int = 5

# MM:
#   Conversion factor from mm to inches.
MM = 1 / 25.4

# NK_DATABASE_FILENAME:
#   The default filename to use for the nk database.
NK_DATABASE_FILENAME: str = "winch_mixed_nk_database.csv"

# WAVELENGTH_RESOLUTION:
#   The resolution to use for the wavelength.
WAVELENGTH_RESOLUTION: float = 1000


@dataclass(frozen=True)
class Layer:
    """
    Represents a layer.

    .. attribute:: material
        The name of the layer.

    .. attribute:: thickness
        The thickness of the layer.
    """

    material: str
    thickness: float


def _sanitise_label(label: str) -> str:
    """
    Sanitise the labels to make the active layer human-readable.

    :param: label:
        The label to sanitise.

    :returns:
        The sanitised label.

    """

    if ":" in label:
        return ACTIVE_LAYER

    return label


def load_stack(stack_name: str) -> list[Layer]:
    """
    Load a stack from an input file provided.

    :param: stack_name:
        The name of the stack file to parse

    :returns: The stack as a `list` of :class:`Layer` instances.

    """

    # If the stack file is not a JSON file, throw an error.
    if not stack_name.endswith(".json"):
        stack_name = f"{stack_name}.json"

    if not os.path.isfile(stack_name):
        raise FileNotFoundError(f"Input file not found: {stack_name}")

    # Otherwise, load the layer information from the file.
    with open(stack_name, "r", encoding="UTF-8") as stack_file:
        return [Layer(**entry) for entry in json.load(stack_file)]


def tmm(
    stack_name: str,
    angles_array: float | list[float] = [0],
    max_wavelength: float = 3000,
    min_wavelength: float = 200,
    nk_database_filename: str = NK_DATABASE_FILENAME,
    *,
    use_julia: bool = False,
    wavelength_resolution: float = WAVELENGTH_RESOLUTION,
):
    """
    Compute the TMM for a given stack name, file, angles, and wavelength limits.

    :param: stack_name:
        The name of the stack file to use.

    :param: angles_array:
        The array of incident light angles to use.

    :param: max_wavelength:
        The maximum wavelength value to use.

    :param: min_wavelength:
        The minimum wavelength value to use.

    :param: nk_database_filename:
        The database filename to use.

    """

    if use_julia:
        raise NotImplementedError("Julia branch not implemented.")

    # Parse the stack.
    stack = load_stack(stack_name)

    # Open the nk database file.
    if not nk_database_filename.endswith(".csv"):
        nk_database_filename = f"{nk_database_filename}.csv"

    if not os.path.isfile(nk_database_filename):
        raise FileNotFoundError(f"No nk-database file found: {nk_database_filename}.")

    with open(nk_database_filename, "r", encoding="UTF-8") as nk_database_file:
        nk_database: pd.DataFrame = pd.read_csv(nk_database_file)

    # Determine the minimum and maximum wavelength values to use.
    max_wavelength = min(
        max_wavelength,
        nk_database[[entry for entry in nk_database.columns if "wavelength" in entry]]
        .max()
        .min(),
    )
    min_wavelength = max(
        min_wavelength,
        nk_database[[entry for entry in nk_database.columns if "wavelength" in entry]]
        .min()
        .max(),
    )

    # Create a wavelength series linearly spaced by the wavelength resolution.
    wavelength_series = linspace(min_wavelength, max_wavelength, wavelength_resolution)

    # Construct a series of refractive indices
    layer_nk_functions: list[callable] = [
        interp1d(
            nk_database[f"{layer.material}_wavelength"].dropna(),
            nk_database[f"{layer.material}_n"].dropna()
            + 1j * nk_database[f"{layer.material}_k"].dropna(),
            kind="quadratic",
        )
        for layer in stack
    ]

    # Compute the transmittance through the stack.
    coh_tmm_data = [
        coh_tmm(
            "s",
            (
                _stack_nk := [1]
                + [nk_function(lambda_vac) for nk_function in layer_nk_functions]
                + [1]
            ),
            (
                _stack_thicknesses := [inf]
                + [layer.thickness for layer in stack]
                + [inf]
            ),
            0,
            lambda_vac,
        )
        for lambda_vac in wavelength_series
    ]
    transmittance = [entry["T"] for entry in coh_tmm_data]
    reflectance = [entry["R"] for entry in coh_tmm_data]
    absorptance = [1 - (entry["R"] + entry["T"]) for entry in coh_tmm_data]

    # Plot the TMM
    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.set_style("whitegrid")
    data = pd.DataFrame(
        {
            "Reflectance": reflectance,
            "Transmittance": transmittance,
            "Absorptance": absorptance,
        }
    )
    data.index = wavelength_series
    sns.lineplot(data, palette=sns.color_palette(["#FDE725", "#21908C", "#440154"]))
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Fraction reflected, transmitted or absorbed")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.gca().tick_params(axis="both", which="major", labelsize=7)
    plt.savefig(
        f"python_tmm_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05
    )

    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.set_style("whitegrid")
    data = pd.DataFrame(
        {
            "Reflectance": reflectance,
            "Transmittance": transmittance,
            "Absorptance": absorptance,
        }
    )
    data.index = wavelength_series
    plt.plot(data.index, data.Absorptance, color="#440154", label="Absorptance")
    plt.plot(
        data.index,
        data.Absorptance + data.Transmittance,
        "--",
        color="#21908C",
        label="Transmittance",
    )
    plt.plot(
        data.index, [1] * len(data.index), "-.", color="#FDE725", label="Reflectance"
    )
    plt.fill_between(
        data.index, [0] * len(data.index), data.Absorptance, color="#440154", alpha=0.3
    )
    plt.fill_between(
        data.index,
        data.Absorptance,
        data.Absorptance + data.Transmittance,
        color="#21908C",
        alpha=0.3,
    )
    plt.fill_between(
        data.index,
        data.Absorptance + data.Transmittance,
        [1] * len(data.index),
        color="#FDE725",
        alpha=0.3,
    )
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Fraction reflected, transmitted or absorbed")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.gca().tick_params(axis="both", which="major", labelsize=7)
    plt.savefig(
        f"python_tmm_stacked_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.set_style("whitegrid")
    data = pd.DataFrame(
        {
            "Reflectance": reflectance,
            "Transmittance": transmittance,
            "Absorptance": absorptance,
        }
    )
    data.index = wavelength_series
    plt.plot(data.index, data.Absorptance, color="#440154", label="Absorptance")
    plt.plot(
        data.index,
        data.Absorptance + data.Transmittance,
        "--",
        color="#21908C",
        label="Transmittance",
    )
    plt.plot(
        data.index, [1] * len(data.index), "-.", color="#FDE725", label="Reflectance"
    )
    plt.fill_between(
        data.index, [0] * len(data.index), data.Absorptance, color="#440154", alpha=0.3
    )
    plt.fill_between(
        data.index,
        data.Absorptance,
        data.Absorptance + data.Transmittance,
        color="#21908C",
        alpha=0.3,
    )
    plt.fill_between(
        data.index,
        data.Absorptance + data.Transmittance,
        [1] * len(data.index),
        color="#FDE725",
        alpha=0.3,
    )
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Fraction reflected, transmitted or absorbed")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.gca().tick_params(axis="both", which="major", labelsize=7)
    plt.savefig(
        f"python_tmm_stacked_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    sns.set_style("ticks")

    absorption: dict[int, list[float]] = {}
    poynting_vector: dict[int, list[float]] = {}
    for wavelength in (
        wavelengths := [
            350,
            550,
            750,
            950,
            1150,
            1250,
            1350,
        ]
    ):
        # Save the absorptance as a function of depth
        coh_tmm_data = coh_tmm("p", _stack_nk, _stack_thicknesses, 0, wavelength)
        depths = linspace(
            -50,
            50 + sum([entry for entry in _stack_thicknesses if entry != inf]),
            num=1000,
        )  # position in structure
        depth_data = [
            position_resolved(
                *find_in_structure_with_inf(_stack_thicknesses, depth), coh_tmm_data
            )
            for depth in depths
        ]
        absorption[wavelength] = [entry["absor"] for entry in depth_data]
        poynting_vector[wavelength] = [entry["poyn"] for entry in depth_data]

        # Plot the absorptance within each layer.
        _data = pd.DataFrame(
            {
                "layer": (_x := [_sanitise_label(entry.material) for entry in stack]),
                "absorptance": 100 * absorp_in_each_layer(coh_tmm_data)[1:-1],
            }
        )

        plt.figure(figsize=(83 * MM, 60 * MM))
        _barplot = sns.barplot(
            _data,
            x=_data.index,
            y="absorptance",
            palette=BACKGROUND_PALETTE,
            hue="layer",
            legend=False,
            ax=(axis := plt.gca()),
        )
        # _barplot.set_yscale("log")
        plt.xticks(plt.xticks()[0], _data.layer)
        plt.xlabel("Layer", fontsize=7)
        plt.ylabel("Absorptance / %", fontsize=7)
        axis.set_ylim(0, 57.5)
        axis.tick_params(axis="both", which="major", labelsize=7)
        plt.savefig(
            f"layerwise_absorptance_unique_{wavelength}_nm_{INDEX}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
        )

        plt.figure(figsize=(83 * MM, 60 * MM))
        _barplot = sns.barplot(
            _data,
            x="layer",
            y="absorptance",
            palette=BACKGROUND_PALETTE,
            hue="layer",
            legend=False,
            ax=(axis := plt.gca()),
        )
        # _barplot.set_yscale("log")
        plt.xlabel("Layer", fontsize=7)
        plt.ylabel("Absorptance / %", fontsize=7)
        axis.set_ylim(0, 57.5)
        axis.tick_params(axis="both", which="major", labelsize=7)
        plt.savefig(
            f"layerwise_absorptance_{wavelength}_nm_{INDEX}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
        )

    absorption_frame = pd.DataFrame(absorption)
    absorption_frame.index = depths

    def _get_hatch(index: int, repeating_index: int) -> str | None:
        """
        Determine the hatching to use.

        :param: index:
            The current index.

        :param: repeating_index:
            The index at which the index should repeat.

        :returns:
            The hatching to use.

        """
        if index < (
            num_colours := (
                repeating_index
                if repeating_index is not None
                else len(sns.color_palette())
            )
        ):
            return None
        if index < 2 * num_colours:
            return "\\\\"
        return "//"

    sns.set_palette(
        sns.cubehelix_palette(
            start=0.6, rot=-0.6, light=0.70, n_colors=len(wavelengths)
        )
    )

    # Plot the absorptance as a function of depth.
    fig = plt.figure(figsize=(171 * MM, 100 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    current_x: float = 0
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        0,
        color=BACKGROUND_PALETTE[0],
        label="Air",
        alpha=0.3,
    )
    for index, layer in enumerate(stack):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(index + 1) % len(BACKGROUND_PALETTE)],
        )
        current_x += layer.thickness

    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = [_sanitise_label(label) for label in labels]
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_with_wavelengths_nm_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_with_wavelengths_nm_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    # Plot the absorptance as a function of depth.
    fig = plt.figure(figsize=(83 * MM, 50 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    current_x: float = 0
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        0,
        color=BACKGROUND_PALETTE[0],
        label="Air",
        alpha=0.3,
    )
    for index, layer in enumerate(stack):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(index + 1) % len(BACKGROUND_PALETTE)],
        )
        current_x += layer.thickness

    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = [_sanitise_label(label) for label in labels]
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_with_wavelengths_nm_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_with_wavelengths_nm_small_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    # plt.show()

    # Plot the absorptance as a function of depth where the glass layers are not shown.
    sns.set_palette(
        [
            "#3455CC",
            "#557BF9",
            "#9FB0FC",
            "#D1C3A7",
            "#FBC412",
            "#FE8224",
            "#E03944",
        ]
    )

    fig = plt.figure(figsize=(171 * MM, 100 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim(
        stack[0].thickness + absorption_frame.index[0],
        absorption_frame.index[-1] - stack[-1].thickness,
    )
    ymin, ymax = plt.ylim()
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        stack[0].thickness,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    color_index: int = 0
    current_x: float = stack[0].thickness
    for index, layer in enumerate(stack[1:-1]):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
        )
        color_index += 1
        current_x += layer.thickness

    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        sum([entry.thickness for entry in stack]) - stack[-1].thickness,
        xmax,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = (
        labels[len(wavelengths) : len(wavelengths) + 3]
        + [ACTIVE_LAYER]
        + labels[len(wavelengths) + 4 :]
    )
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_no_glass_with_wavelengths_nm_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_no_glass_with_wavelengths_nm_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    fig = plt.figure(figsize=(83 * MM, 50 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim(
        stack[0].thickness + absorption_frame.index[0],
        absorption_frame.index[-1] - stack[-1].thickness,
    )
    ymin, ymax = plt.ylim()
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        stack[0].thickness,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    color_index: int = 0
    current_x: float = stack[0].thickness
    for index, layer in enumerate(stack[1:-1]):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
        )
        color_index += 1
        current_x += layer.thickness

    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        sum([entry.thickness for entry in stack]) - stack[-1].thickness,
        xmax,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = (
        labels[len(wavelengths) : len(wavelengths) + 3]
        + [ACTIVE_LAYER]
        + labels[len(wavelengths) + 4 :]
    )
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_no_glass_with_wavelengths_nm_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_no_glass_with_wavelengths_nm_small_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    # plt.show()

    # Plot the spectra but ignore the next layer (ITO).
    fig = plt.figure(figsize=(171 * MM, 100 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim(
        sum([entry.thickness for entry in stack[:2]]) + absorption_frame.index[0],
        absorption_frame.index[-1] - stack[-1].thickness,
    )
    ymin, ymax = plt.ylim()
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        (current_x := sum([entry.thickness for entry in stack[:2]])),
        color=BACKGROUND_PALETTE[1],
        label=stack[1].material,
        alpha=0.3,
    )
    color_index: int = 1
    for index, layer in enumerate(stack[2:-1]):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
        )
        color_index += 1
        current_x += layer.thickness

    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        sum([entry.thickness for entry in stack]) - stack[-1].thickness,
        xmax,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = (
        labels[len(wavelengths) : len(wavelengths) + 2]
        + [ACTIVE_LAYER]
        + labels[len(wavelengths) + 3 :]
    )
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_no_glass_or_ito_with_wavelengths_nm_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_no_glass_or_ito_with_wavelengths_nm_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    # Plot the spectra but ignore the next layer (ITO).
    fig = plt.figure(figsize=(83 * MM, 50 * MM))
    axis = plt.gca()
    sns.lineplot(absorption_frame, ax=axis)
    plt.xlabel("Depth / mm")
    plt.ylabel("Absorption / %")

    # Add the layer information.
    xmin, xmax = plt.xlim(
        sum([entry.thickness for entry in stack[:2]]) + absorption_frame.index[0],
        absorption_frame.index[-1] - stack[-1].thickness,
    )
    ymin, ymax = plt.ylim()
    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        xmin,
        (current_x := sum([entry.thickness for entry in stack[:2]])),
        color=BACKGROUND_PALETTE[1],
        label=stack[1].material,
        alpha=0.3,
    )
    color_index: int = 1
    for index, layer in enumerate(stack[2:-1]):
        axis.fill_betweenx(
            linspace(ymin, ymax, 1000),
            current_x,
            current_x + layer.thickness,
            color=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
            hatch=_get_hatch(index, len(BACKGROUND_PALETTE)),
            label=layer.material,
            alpha=0.25,
            edgecolor=BACKGROUND_PALETTE[(color_index + 1) % len(BACKGROUND_PALETTE)],
        )
        color_index += 1
        current_x += layer.thickness

    axis.fill_betweenx(
        linspace(ymin, ymax, 1000),
        sum([entry.thickness for entry in stack]) - stack[-1].thickness,
        xmax,
        color=BACKGROUND_PALETTE[0],
        label=stack[0].material,
        alpha=0.3,
    )
    # axis.fill_betweenx(linspace(ymin, ymax, 1000), current_x, xmax, 0, color="C0", alpha=0.3)
    axis.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    legend1 = plt.legend(
        handles[: len(wavelengths)],
        labels[: len(wavelengths)],
        loc=2,
        fontsize=7,
        title="Wavelength / nm",
        title_fontsize=7,
    )
    material_labels = (
        labels[len(wavelengths) : len(wavelengths) + 2]
        + [ACTIVE_LAYER]
        + labels[len(wavelengths) + 3 :]
    )
    plt.legend(
        handles[len(wavelengths) :],
        material_labels,
        loc=1,
        fontsize=7,
        ncols=1,
        title="Layer",
        title_fontsize=7,
    )
    axis.add_artist(legend1)

    plt.savefig(
        f"absorption_no_glass_or_ito_with_wavelengths_nm_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.savefig(
        f"absorption_no_glass_or_ito_with_wavelengths_nm_small_{INDEX}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        dpi=1500,
    )

    # plt.show()

    # Compute the absorption within the active layer as a function of wavelength.

    # Determine the index of the active layer in the stack.
    active_layer_index: int = {
        _sanitise_label(layer.material): index for index, layer in enumerate(stack)
    }[ACTIVE_LAYER]

    # Run the computation to determine the absorptance for each wavelength.
    layerwise_absorption: dict[int, list[float]] = {}
    for wavelength in tqdm(
        wavelength_series, desc="Active-layer absorption calculation", unit="nm"
    ):
        # Save the absorptance as a function of depth
        coh_tmm_data = coh_tmm("p", _stack_nk, _stack_thicknesses, 0, wavelength)
        layerwise_absorption[wavelength] = absorp_in_each_layer(coh_tmm_data)[1:-1]

    layerwise_absorption_frame = pd.DataFrame(layerwise_absorption).transpose()
    layerwise_absorption_frame.columns = pd.Index(
        [_sanitise_label(layer.material) for layer in stack]
    )

    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        layerwise_absorption_frame, palette=BACKGROUND_PALETTE, ax=(axis := plt.gca())
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_layers_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        layerwise_absorption_frame.mul(data.Absorptance, axis=0),
        palette=BACKGROUND_PALETTE,
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_layers_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        layerwise_absorption_frame, palette=BACKGROUND_PALETTE, ax=(axis := plt.gca())
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_layers_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        layerwise_absorption_frame.mul(data.Absorptance, axis=0),
        palette=BACKGROUND_PALETTE,
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_layers_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    sns.set_palette(sns.color_palette([BACKGROUND_PALETTE[active_layer_index]]))
    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
        label=ACTIVE_LAYER,
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_active_layer_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame.mul(data.Absorptance, axis=0)[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
        label=ACTIVE_LAYER,
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_active_layer_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
        label=ACTIVE_LAYER,
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_active_layer_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame.mul(data.Absorptance, axis=0)[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
        label=ACTIVE_LAYER,
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_active_layer_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    sns.set_palette(sns.color_palette(["#440154"]))
    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_active_layer_purple_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame.mul(data.Absorptance, axis=0)[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_active_layer_purple_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.index,
        y=layerwise_absorption_frame[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)

    plt.savefig(
        f"relative_absorptance_in_active_layer_purple_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.show()

    plt.figure(figsize=(83 * MM, 60 * MM))
    sns.lineplot(
        x=layerwise_absorption_frame.mul(data.Absorptance, axis=0).index,
        y=layerwise_absorption_frame[ACTIVE_LAYER],
        ax=(axis := plt.gca()),
    )
    axis.set_xlabel("Wavelength / nm", fontsize=7)
    axis.set_ylabel("Absorptance", fontsize=7)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.legend(fontsize=7)
    axis.set_ylim(0, 1)

    plt.savefig(
        f"absolute_absorptance_in_active_layer_purple_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.show()

    return

    for depth in depths:
        layer, d_in_layer = find_in_structure_with_inf(_stack_thicknesses, depth)
        data = position_resolved(layer, d_in_layer, coh_tmm_data)
        poyn.append(data["poyn"])
        absor.append(data["absor"])
    # convert data to numpy arrays for easy scaling in the plot
    poyn = array(poyn)
    absor = array(absor)
    plt.figure()
    plt.plot(ds, poyn, "blue", ds, 200 * absor, "purple")
    plt.xlabel("depth (nm)")
    plt.ylabel("AU")
    plt.title("Local absorption (purple), Poynting vector (blue)")
    # plt.show()

    import pdb

    pdb.set_trace()

    # d_list = [inf, 100, 300, inf] #in nm
    # n_list = [1, 2.2+0.2j, 3.3+0.3j, 1]
    # th_0 = pi/4
    # lam_vac = 400
    # pol = 'p'
    # coh_tmm_data = coh_tmm(pol, n_list, d_list, th_0, lam_vac)

    # ds = linspace(-50, 400, num=1000) #position in structure
    # poyn = []
    # absor = []
    # for d in ds:
    #     layer, d_in_layer = find_in_structure_with_inf(d_list, d)
    #     data = position_resolved(layer, d_in_layer, coh_tmm_data)
    #     poyn.append(data['poyn'])
    #     absor.append(data['absor'])
    # # convert data to numpy arrays for easy scaling in the plot
    # poyn = array(poyn)
    # absor = array(absor)
    # plt.figure()
    # plt.plot(ds, poyn, 'blue', ds, 200*absor, 'purple')
    # plt.xlabel('depth (nm)')
    # plt.ylabel('AU')
    # plt.title('Local absorption (purple), Poynting vector (blue)')
    # # plt.show()


if __name__ == "__main__":
    tmm("stack_reversed")
