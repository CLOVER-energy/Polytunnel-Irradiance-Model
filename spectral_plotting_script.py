#!/usr/bin/python3.12
########################################################################################
# Spectral plotting script                                                             #
# Script for plotting the various solar spectral.
########################################################################################

import os
import sys

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pvlib
import seaborn as sns

from matplotlib import rc, rcParams
from scipy import constants

# Plotting context
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 7

# INDEX:
#   Index used for plotting code.
INDEX: int = 3

# MM:
#   Conversion factor from mm to inches.
MM = 1 / 25.4

# PAR_LOWERBOUND:
#   The lowerbound wavelength for PAR.
PAR_LOWERBOUND: float = 400

# PAR_UPPERBOUND:
#   The upperbound wavelength for PAR.
PAR_UPPERBOUND: float = 700


def main(args: list[Any]) -> None:
    """
    Main plotting function.

    :param: args:
        The unparsed command-line arguments.

    """

    # Set the colour palette.
    sns.set_palette(
        # ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#0041C8"]
        # [
        #     "#423252",
        #     "#4A688B",
        #     "#779FB1",
        #     "#36C7B8",
        #     "#FBC412",
        #     "#e04606",
        #     "#423252",
        #     "#49678B",
        #     "#36C7B8",
        #     "#FE8224",
        #     "#E03944",
        # ],
        sns.color_palette("viridis", n_colors=6).as_hex()
        + [
            "#423252",
            "#49678B",
            "#36C7B8",
            "#FE8224",
            "#E03944",
        ],
    )
    palette = sns.color_palette()

    # Load the reference spectrum.
    reference_spectra = pvlib.spectrum.get_reference_spectra()

    # Load the cloudy-day spectra.
    with open(
        "brecl_spectra_on_typical_days.csv", "r", encoding="UTF-8"
    ) as brecl_spectra_file:
        brecl_spectra = pd.read_csv(brecl_spectra_file)

    brecl_plot = brecl_spectra.transpose()
    for index in range(8):
        brecl_plot.pop(index)

    brecl_plot = brecl_plot.transpose()
    brecl_plot = brecl_plot.astype(float)
    brecl_plot = brecl_plot.set_index("wavelength")

    sns.set_style("ticks")
    sns.set_context("paper")

    # Load the data for the chlorophyl and bee response.
    with open(
        os.path.join((_austin_dir := "austin_spectra"), "blue_wavelength.csv"),
        "r",
        encoding="UTF-8",
    ) as file_handler:
        blue_data = pd.read_csv(file_handler, index_col=0)

    with open(
        os.path.join(_austin_dir, "uv_wavelength.csv"), "r", encoding="UTF-8"
    ) as file_handler:
        uv_data = pd.read_csv(file_handler, index_col=0)

    with open(
        os.path.join(_austin_dir, "green_wavelength.csv"), "r", encoding="UTF-8"
    ) as file_handler:
        green_data = pd.read_csv(file_handler, index_col=0)

    with open(
        os.path.join(_austin_dir, "chlorophyl_b_wavelength.csv"), "r", encoding="UTF-8"
    ) as file_handler:
        chlorophyl_b_data = pd.read_csv(file_handler, index_col=0)

    with open(
        os.path.join(_austin_dir, "chlorophyl_a_wavelength.csv"), "r", encoding="UTF-8"
    ) as file_handler:
        chlorophyl_a_data = pd.read_csv(file_handler, index_col=0)

    plt.figure(figsize=(83 * MM, 60 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_plot.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND, brecl_plot.index[-1], hatch="//", color="#CDCDCD", alpha=0.3
    )
    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]

    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]],
        labels[:3] + list(reversed(labels[3:6])) + [handles[6]],
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_small_sans_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # plt.show()

    plt.figure(figsize=(83 * MM, 60 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_plot.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND, brecl_plot.index[-1], hatch="//", color="#CDCDCD", alpha=0.3
    )
    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]

    axis_right = axis.twinx()
    axis_right.plot(
        uv_data["λ / nm"] * 10**9,
        uv_data.response,
        "--",
        label="UV cone-cell response",
        color="C6",
    )
    axis_right.plot(
        blue_data["λ / nm"] * 10**9,
        blue_data.response,
        "--",
        label="Blue cone-cell response",
        color="C7",
    )
    axis_right.plot(
        green_data["λ / nm"] * 10**9,
        green_data.response,
        "--",
        label="Green cone-cell response",
        color="C8",
    )
    axis_right.plot(chlorophyl_a_data, "-.", label="Chlorophyll-a response", color="C9")
    axis_right.plot(
        chlorophyl_b_data, "-.", label="Chlorophyll-b response", color="C10"
    )
    axis_right.set_ylabel("Normalised response")
    axis.legend().remove()
    axis_right.legend().remove()

    right_handles, right_labels = axis_right.get_legend_handles_labels()

    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]] + right_handles,
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]] + right_labels,
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_small_with_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # plt.show()

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_plot.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND, brecl_plot.index[-1], hatch="//", color="#CDCDCD", alpha=0.3
    )
    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]],
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]],
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_large_sans_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.show()

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_plot.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND, brecl_plot.index[-1], hatch="//", color="#CDCDCD", alpha=0.3
    )
    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]

    axis_right = axis.twinx()
    axis_right.plot(
        uv_data["λ / nm"] * 10**9,
        uv_data.response,
        "--",
        label="UV cone-cell response",
        color="C6",
    )
    axis_right.plot(
        blue_data["λ / nm"] * 10**9,
        blue_data.response,
        "--",
        label="Blue cone-cell response",
        color="C7",
    )
    axis_right.plot(
        green_data["λ / nm"] * 10**9,
        green_data.response,
        "--",
        label="Green cone-cell response",
        color="C8",
    )
    axis_right.plot(chlorophyl_a_data, "-.", label="Chlorophyll-a response", color="C9")
    axis_right.plot(
        chlorophyl_b_data, "-.", label="Chlorophyll-b response", color="C10"
    )
    axis_right.set_ylabel("Normalised response")
    axis.legend().remove()
    axis_right.legend().remove()

    right_handles, right_labels = axis_right.get_legend_handles_labels()

    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]] + right_handles,
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]] + right_labels,
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_large_with_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.show()

    # Plot the spectra with photon flux rather than energy.
    reference_energy_series = (
        constants.h * constants.c / (reference_spectra.index * 10 ** (-9))
    )
    reference_spectra_flux = (
        reference_spectra.divide(reference_energy_series, axis=0)
        / (10**4 * constants.N_A)  # Convert to micro-moles per cm2
        * 10**6
    )

    brecl_energy_series = constants.h * constants.c / (brecl_plot.index * 10 ** (-9))
    brecl_spectra_flux = (
        brecl_plot.divide(brecl_energy_series, axis=0)
        / (10**4 * constants.N_A)  # Convert to micro-moles per cm2
        * 10**6
    )

    plt.figure(figsize=(83 * MM, 60 * MM))

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_spectra_flux.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND,
        brecl_spectra_flux.index[-1],
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
    )
    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]

    axis_right = axis.twinx()
    axis_right.plot(
        uv_data["λ / nm"] * 10**9,
        uv_data.response,
        "--",
        label="UV cone-cell response",
        color="C6",
    )
    axis_right.plot(
        blue_data["λ / nm"] * 10**9,
        blue_data.response,
        "--",
        label="Blue cone-cell response",
        color="C7",
    )
    axis_right.plot(
        green_data["λ / nm"] * 10**9,
        green_data.response,
        "--",
        label="Green cone-cell response",
        color="C8",
    )
    axis_right.plot(chlorophyl_a_data, "-.", label="Chlorophyll-a response", color="C9")
    axis_right.plot(
        chlorophyl_b_data, "-.", label="Chlorophyll-b response", color="C10"
    )
    axis_right.set_ylabel("Normalised response")
    axis.legend().remove()
    axis_right.legend().remove()

    right_handles, right_labels = axis_right.get_legend_handles_labels()

    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]] + right_handles,
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]] + right_labels,
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
    axis_right.set_ylabel("Normalised photo-response intensity")

    plt.savefig(
        f"brecl_solar_flux_small_avec_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(83 * MM, 60 * MM))

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_spectra_flux.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
        label="Photosynthetically-inactive region",
    )
    axis.axvspan(
        PAR_UPPERBOUND,
        brecl_spectra_flux.index[-1],
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
    )
    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]],
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]] + [labels[6]],
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
    axis_right.set_ylabel("Normalised photo-response intensity")

    plt.savefig(
        f"brecl_solar_flux_small_sans_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_spectra_flux.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        label="Photosynthetically-inactive region",
        alpha=0.3,
    )
    axis.axvspan(
        PAR_UPPERBOUND,
        brecl_spectra_flux.index[-1],
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
    )
    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]

    axis_right = axis.twinx()
    axis_right.plot(
        uv_data["λ / nm"] * 10**9,
        uv_data.response,
        "--",
        label="UV cone-cell response",
        color="C6",
    )
    axis_right.plot(
        blue_data["λ / nm"] * 10**9,
        blue_data.response,
        "--",
        label="Blue cone-cell response",
        color="C7",
    )
    axis_right.plot(
        green_data["λ / nm"] * 10**9,
        green_data.response,
        "--",
        label="Green cone-cell response",
        color="C8",
    )
    axis_right.plot(chlorophyl_a_data, "-.", label="Chlorophyll-a response", color="C9")
    axis_right.plot(
        chlorophyl_b_data, "-.", label="Chlorophyll-b response", color="C10"
    )
    axis_right.set_ylabel("Normalised response")
    axis.legend().remove()
    axis_right.legend().remove()

    right_handles, right_labels = axis_right.get_legend_handles_labels()

    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]] + right_handles,
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]] + right_labels,
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
    axis_right.set_ylabel("Normalised photo-response intensity")

    plt.savefig(
        f"brecl_solar_flux_large_avec_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:6]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    ylim = plt.ylim()
    axis.axvspan(
        brecl_spectra_flux.index[0],
        PAR_LOWERBOUND,
        hatch="//",
        color="#CDCDCD",
        label="Photosynthetically-inactive region",
        alpha=0.3,
    )
    axis.axvspan(
        PAR_UPPERBOUND,
        brecl_spectra_flux.index[-1],
        hatch="//",
        color="#CDCDCD",
        alpha=0.3,
    )
    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    plt.ylim(*ylim)

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:6])) + [handles[6]],
        labels[:3] + list(reversed(labels[3:6])) + [labels[6]],
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    axis.set_ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})
    axis_right.set_ylabel("Normalised photo-response intensity")

    plt.savefig(
        f"brecl_solar_flux_large_sans_response_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
