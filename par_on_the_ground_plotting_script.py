#!/usr/bin/python3

import argparse
import collections
import enum
import itertools
import json
import math
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import pdb
import re
import sys

from matplotlib import rc
from matplotlib import ticker
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm

# DPI:
#   The DPI to use when saving figures.
DPI: int = 400

# INDEX:
#   Variable used for plotting to ensure non-overwriting behaviour when saving plots
INDEX: int = 21

# MM:
#   Conversion factor from mm to inches.
MM = 1 / 25.4

# MONTHS:
#   A list of names of months
MONTHS: list[str] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

PURPLE_GREEN_COLOURS: list[str] = [
    "#003D30",
    "#005745",
    "#00735C",
    "#009175",
    "#00AD8E",
    "#00CBA7",
    "#00EBC1",
    "#86FFDE",
    "#FFCCFE",
    "#FF92FD",
    "#FF3CFE",
    "#DA00FD",
    "#A700FC",
    "#8400CD",
    "#8400CD",
    "#65019F",
    "#450270",
]

# Plotting context
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 7


TOTEX_HEADER: str = "Other TOTEX"

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
sns.set_context("paper")
sns.set_style("whitegrid")

# Set custom color-blind colormap
colorblind_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

sns.set_palette(colorblind_palette)

thesis_palette = sns.color_palette(
    [
        "#E04606",
        "#F9A130",
        "#FCB919",  # SDG 7
        "#EFF2DD",
        "#27BFE6",  # SDG 6
        "#144E56",
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
    ]
)

categorical_colourblind_palette = sns.color_palette(
    [
        "#77AADD",
        "#99DDFF",
        "#44BB99",
        "#BBCC33",
        "#AAA000",
        "#EEDD88",
        "#EE8866",
        "#FFAABB",
        "#DDDDDD",
    ]
)

sns.blend_palette(["#36C7B8", "#423252"], n_colors=50)

# Figure size
# fig, axes = plt.subplots(2, 2, figsize=(48 / 5, 32 / 5))

# fig = plt.figure(figsize=(48 / 5, 32 / 5))

sns.set_style("ticks")
sns.set_palette(categorical_colourblind_palette)

# Open the plotting data for the PAR on the ground.
file_regex = re.compile(
    r"(?P<irradiance>.*)sky_(?P<data_type>[^_]*)_flux_umol_m2_(?P<polytunnel>[a-z_]*)_"
)
data: list[pd.DataFrame] = []
irradiances: set[str] = set()
for filename in os.listdir("."):
    match = file_regex.match(filename)
    if match is None:
        continue

    if match.group("data_type") != "mean":
        continue

    with open(filename, "r", encoding="UTF-8") as file_handler:
        this_data = pd.read_csv(file_handler, index_col=0)

    irradiances.add((irradiance := match.group("irradiance").replace("_", "")))
    column_name: str = (
        (f"{match.group("polytunnel").replace("_short", "")}_{irradiance}")
        .replace("circular_", "")
        .replace("_", " ")
        .capitalize()
    )

    this_data.columns = pd.Index([column_name])
    data.append(this_data)

# Assemble data based on timeframe.
timeframes = {len(entry) for entry in data}

concatenated_data = {
    timeframe: pd.concat([entry for entry in data if len(entry) == timeframe], axis=1)
    for timeframe in timeframes
}

# Plot these
for timeframe, data in concatenated_data.items():
    # Filter the data based on the incoming irradiance.
    for irradiance in irradiances:
        plotting_data = data.transpose()[
            [irradiance in entry for entry in data.columns]
        ].transpose()
        # plotting_data = plotting_data.reindex(sorted(plotting_data.columns), axis=1)

        fig = plt.figure(figsize=(171 * MM, 120 * MM))
        axis = plt.gca()

        sns.lineplot(plotting_data)

        axis.legend(fontsize=7)

        plt.xlabel("Hour")
        plt.ylabel(r"PPFD ($\Phi_{\rm{PAR}}$) / $\mu$mol / m$^2$")

        plt.show()
