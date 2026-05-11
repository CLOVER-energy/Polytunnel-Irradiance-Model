#!/usr/bin/python3.12
########################################################################################
# Spectral plotting script                                                             #
# Script for plotting the various solar spectral.
########################################################################################

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
INDEX: int = 1

# MM:
#   Conversion factor from mm to inches.
MM = 1 / 25.4


def main(args: list[Any]) -> None:
    """
    Main plotting function.

    :param: args:
        The unparsed command-line arguments.

    """

    # Set the colour palette.
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

    plt.figure(figsize=(83 * MM, 60 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:])),
        labels[:3] + list(reversed(labels[3:])),
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra, palette=reversed(palette[3:]))
    sns.lineplot(brecl_plot / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    plt.xlim(
        max(brecl_plot.index[0], reference_spectra.index[0]),
        min(brecl_plot.index[-1], reference_spectra.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:])),
        labels[:3] + list(reversed(labels[3:])),
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Irradiance / W/m$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_spectra_large_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    # plt.show()

    import pdb

    pdb.set_trace()

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

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:])),
        labels[:3] + list(reversed(labels[3:])),
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_flux_small_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.figure(figsize=(171 * MM, 120 * MM))

    sns.lineplot(reference_spectra_flux, palette=reversed(palette[3:]))
    sns.lineplot(brecl_spectra_flux / 1000, palette=palette)

    axis = plt.gca()
    axis.tick_params(axis="both", which="major", labelsize=7)

    plt.xlim(
        max(brecl_spectra_flux.index[0], reference_spectra_flux.index[0]),
        min(brecl_spectra_flux.index[-1], reference_spectra_flux.index[-1]),
    )

    handles, labels = axis.get_legend_handles_labels()
    labels = [entry.capitalize().replace("_", " ") for entry in labels]
    plt.legend(
        handles[:3] + list(reversed(handles[3:])),
        labels[:3] + list(reversed(labels[3:])),
        title="Spectrum",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
    )
    plt.xlabel("Wavelength ($\lambda$) / nm", fontdict={"size": 7})
    plt.ylabel("Photon flux ($\Phi$) / $\mu$mol/cm$^2$nm", fontdict={"size": 7})

    plt.savefig(
        f"brecl_solar_flux_large_{INDEX}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
