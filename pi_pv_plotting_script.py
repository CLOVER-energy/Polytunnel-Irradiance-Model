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
INDEX: int = 30

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
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 5})
sns.set_context("paper", rc={"font.size": 5, "axes.titlesize": 5, "axes.labelsize": 5})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from matplotlib import rcParams

# Plotting context
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
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

pi_pv_palette = sns.color_palette(
    [
        "#423252",
        "#0a77aa",
        "#36c7b8",
        "#fbc412",
        "#e04606",
    ]
)
sns.set_palette(pi_pv_palette)

# Figure size
# fig, axes = plt.subplots(2, 2, figsize=(48 / 5, 32 / 5))

# fig = plt.figure(figsize=(48 / 5, 32 / 5))

with open(
    "predicted_day_total_map_0.55_circular_control_short_2024-03-01T00_00_00Z_2024-10-31T23_59_59Z.csv",
    "r",
) as f:
    control_total = pd.read_csv(f, index_col=0)

with open(
    "predicted_day_total_map_0.55_circular_wide_short_2024-03-01T00_00_00Z_2024-10-31T23_59_59Z.csv",
    "r",
) as f:
    wide_total = pd.read_csv(f, index_col=0)

with open(
    "predicted_day_total_map_0.55_circular_narrow_short_2024-03-01T00_00_00Z_2024-10-31T23_59_59Z.csv",
    "r",
) as f:
    narrow_total = pd.read_csv(f, index_col=0)

control_mean = control_total.mean(axis=1).reset_index(drop=True)
wide_mean = wide_total.mean(axis=1).reset_index(drop=True)
narrow_mean = narrow_total.mean(axis=1).reset_index(drop=True)

control_daily = control_mean.groupby(control_mean.index // 24).sum()
wide_daily = wide_mean.groupby(wide_mean.index // 24).sum()
narrow_daily = narrow_mean.groupby(narrow_mean.index // 24).sum()

frame = pd.DataFrame(
    {
        "control": control_daily,
        "wide": wide_daily,
        "narrow": narrow_daily,
    }
)
frame["day"] = frame.index


def _month_from_day(day: int) -> int:
    if day <= 31:
        return 3
    if day <= 61:
        return 4
    if day <= 92:
        return 5
    if day <= 122:
        return 6
    if day <= 153:
        return 7
    if day <= 184:
        return 8
    if day <= 214:
        return 9
    return 10


def _month_length_from_day(day: int) -> int:
    if day <= 31:
        return 31
    if day <= 61:
        return 30
    if day <= 92:
        return 31
    if day <= 122:
        return 30
    if day <= 153:
        return 31
    if day <= 184:
        return 31
    if day <= 214:
        return 30
    return 31


frame["month"] = [_month_from_day(entry) for entry in frame["day"]]
frame["month_length"] = [_month_length_from_day(entry) for entry in frame["day"]]

# Divide by the number of days in the month to get a daily average
frame["control"] /= frame["month_length"]
frame["narrow"] /= frame["month_length"]
frame["wide"] /= frame["month_length"]
frame.pop("day")
frame.pop("month_length")
mean_frame = frame.groupby("month").mean()
std_frame = frame.groupby("month").std()

sns.set_style("ticks")
sns.set_context("notebook")

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

# Bar plot
axis = mean_frame.plot(
    kind="bar", figsize=(171 * MM, 100 * MM), yerr=std_frame.T.values
)
axis.set_xlabel("Month")
axis.set_xticklabels(
    ["March", "Apr", "May", "June", "July", "August", "September", "October"],
    rotation=45,
)

axis.set_ylabel("Mean daily irradiance / Wh/m$^2$-day")
handles, labels = axis.get_legend_handles_labels()
plt.legend(
    handles, ["No PV", '"Sparse" flexible PV', '"Dense" flexible PV'], fontsize=7
)
axis.tick_params(axis="both", which="major", labelsize=7)

axis.set_ylim(None, 140)

plt.savefig(
    f"par_comparison_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05
)
plt.show()

# Light-sautration point calculation
strawberry_lsp: float = 304

control_lsp = strawberry_lsp * (control_mean >= strawberry_lsp) + control_mean * (
    control_mean <= strawberry_lsp
)
wide_lsp = strawberry_lsp * (wide_mean >= strawberry_lsp) + wide_mean * (
    wide_mean <= strawberry_lsp
)
narrow_lsp = strawberry_lsp * (narrow_mean >= strawberry_lsp) + narrow_mean * (
    narrow_mean <= strawberry_lsp
)

control_daily = control_lsp.groupby(control_lsp.index // 24).sum()
wide_daily = wide_lsp.groupby(wide_lsp.index // 24).sum()
narrow_daily = narrow_lsp.groupby(narrow_lsp.index // 24).sum()

frame = pd.DataFrame(
    {
        "control": control_daily,
        "wide": wide_daily,
        "narrow": narrow_daily,
    }
)
frame["day"] = frame.index

frame["month"] = [_month_from_day(entry) for entry in frame["day"]]
frame["month_length"] = [_month_length_from_day(entry) for entry in frame["day"]]

# Divide by the number of days in the month to get a daily average
frame["control"] /= frame["month_length"]
frame["narrow"] /= frame["month_length"]
frame["wide"] /= frame["month_length"]
frame.pop("day")
frame.pop("month_length")
mean_frame = frame.groupby("month").mean()
std_frame = frame.groupby("month").std()

sns.set_style("ticks")
sns.set_context("notebook")

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

# Bar plot
axis = mean_frame.plot(
    kind="bar", figsize=(171 * MM, 100 * MM), yerr=std_frame.T.values
)
axis.set_xlabel("Month")
axis.set_xticklabels(
    ["March", "Apr", "May", "June", "July", "August", "September", "October"],
    rotation=45,
)

axis.set_ylabel("Mean light-saturated irradiance / Wh/m$^2$-day")
handles, labels = axis.get_legend_handles_labels()
plt.legend(
    handles, ["No PV", '"Sparse" flexible PV', '"Dense" flexible PV'], fontsize=7
)
axis.tick_params(axis="both", which="major", labelsize=7)

axis.set_ylim(None, 140)

plt.savefig(
    f"lsp_par_comparison_{INDEX}.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

##############################################################################
# Plot with the irradiance behind to show the PAR and irradiance information #
##############################################################################

narrow_daily = narrow_mean.groupby(narrow_mean.index // 24).sum()
wide_daily = wide_mean.groupby(wide_mean.index // 24).sum()
control_daily = control_mean.groupby(control_mean.index // 24).sum()

frame = pd.DataFrame(
    {
        "control": control_daily,
        "wide": wide_daily,
        "narrow": narrow_daily,
    }
)
frame["day"] = frame.index

frame["month"] = [_month_from_day(entry) for entry in frame["day"]]
frame["month_length"] = [_month_length_from_day(entry) for entry in frame["day"]]

# Divide by the number of days in the month to get a daily average
frame["control"] /= frame["month_length"]
frame["narrow"] /= frame["month_length"]
frame["wide"] /= frame["month_length"]
frame.pop("day")
frame.pop("month_length")
mean_frame = frame.groupby("month").mean()
std_frame = frame.groupby("month").std()

sns.set_style("ticks")
sns.set_context("notebook")

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})
sns.set_style("ticks")

# Bar plot
axis = mean_frame.plot(
    alpha=0.5, hatch="\\\\", kind="bar", figsize=(171 * MM, 100 * MM)
)
axis.set_xlabel("Month")
axis.set_xticklabels(
    ["March", "Apr", "May", "June", "July", "August", "September", "October"],
    rotation=45,
)

# plt.savefig(f"par_comparison_{INDEX}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)
# plt.show()

# Light-sautration point calculation
strawberry_lsp: float = 304

control_lsp = strawberry_lsp * (control_mean >= strawberry_lsp) + control_mean * (
    control_mean <= strawberry_lsp
)
wide_lsp = strawberry_lsp * (wide_mean >= strawberry_lsp) + wide_mean * (
    wide_mean <= strawberry_lsp
)
narrow_lsp = strawberry_lsp * (narrow_mean >= strawberry_lsp) + narrow_mean * (
    narrow_mean <= strawberry_lsp
)

control_daily = control_lsp.groupby(control_lsp.index // 24).sum()
wide_daily = wide_lsp.groupby(wide_lsp.index // 24).sum()
narrow_daily = narrow_lsp.groupby(narrow_lsp.index // 24).sum()

frame = pd.DataFrame(
    {
        "control": control_daily,
        "wide": wide_daily,
        "narrow": narrow_daily,
    }
)
frame["day"] = frame.index

frame["month"] = [_month_from_day(entry) for entry in frame["day"]]
frame["month_length"] = [_month_length_from_day(entry) for entry in frame["day"]]

# Divide by the number of days in the month to get a daily average
frame["control"] /= frame["month_length"]
frame["narrow"] /= frame["month_length"]
frame["wide"] /= frame["month_length"]
frame.pop("day")
frame.pop("month_length")
mean_frame = frame.groupby("month").mean()
std_frame = frame.groupby("month").std()

sns.set_style("ticks")
sns.set_context("notebook")

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 7})
sns.set_context(
    "notebook", rc={"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7}
)
sns.set_style("ticks")

# Bar plot
axis = mean_frame.plot(
    ax=axis, kind="bar", figsize=(171 * MM, 100 * MM), yerr=std_frame.T.values
)
axis.set_xlabel("Month")
axis.set_xticklabels(
    ["March", "Apr", "May", "June", "July", "August", "September", "October"],
    rotation=45,
)

axis.set_ylabel("Mean light-saturated irradiance / Wh/m$^2$-day")
handles, labels = axis.get_legend_handles_labels()
plt.legend(
    handles,
    [
        "No PV ($G$)",
        '"Sparse" PV ($G$)',
        '"Dense" PV ($G$)',
        "No PV (PAR)",
        '"Sparse" PV (PAR)',
        '"Dense" PV (PAR)',
    ],
    fontsize=7,
    ncols=2,
    loc="upper right",
)
axis.tick_params(axis="both", which="major", labelsize=7)

axis.set_ylim(None, 140)

plt.savefig(
    f"lsp_par_with_shading_comparison_{INDEX}.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

#######################################################################
# Plot the irradiance and PAR for the Summer solstice, June 20th 2024 #
#######################################################################

import datetime

days_to_solstice = (datetime.datetime(2024, 6, 20) - datetime.datetime(2024, 3, 1)).days
segments_to_solstice = days_to_solstice * 24

index: int = 475
solstice_control_475 = control_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_wide_475 = wide_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_narrow_475 = narrow_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]

index: int = 425
solstice_control_425 = control_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_wide_425 = wide_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_narrow_425 = narrow_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]

index: int = 275
solstice_control_275 = control_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_wide_275 = wide_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_narrow_275 = narrow_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]

index: int = 0
solstice_control_0 = control_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_wide_0 = wide_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]
solstice_narrow_0 = narrow_total[f"{index}"][
    segments_to_solstice : segments_to_solstice + 24
]

# Plot the irradiance, capped off by the PAR
plt.figure(figsize=(171 * MM, 100 * MM))

sns.lineplot(solstice_control_475, label="No PV", color="C0")
sns.lineplot(solstice_wide_475, label='"Sparse" PV', color="C1")
sns.lineplot(solstice_narrow_475, label='"Dense" PV', color="C2")


def _interp_control(x: list[float]) -> list[float]:
    return np.interp(x, solstice_control_475.index, solstice_control_475.values)


(axis := plt.gca()).axhline(strawberry_lsp, dashes=(4, 4), color="C0", label="LSP")

x_array = np.linspace(0, 23, 1440)
axis.fill_between(
    x_array,
    [0] * 1440,
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_control_475.values)
    ],
    alpha=0.1,
    color="C0",
    hatch="//",
)
axis.fill_between(
    x_array,
    [0] * 1440,
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_wide_475.values)
    ],
    alpha=0.1,
    color="C1",
    hatch="//",
)
axis.fill_between(
    x_array,
    [0] * 1440,
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_narrow_475.values)
    ],
    alpha=0.1,
    color="C2",
    hatch="//",
)

axis.fill_between(
    x_array[:900],
    np.interp(x_array, range(24), solstice_narrow_475.values)[:900],
    np.interp(x_array, range(24), solstice_control_475.values)[:900],
    alpha=0.7,
    color="#ABABAB",
    hatch="//",
    label="PAR above LSP",
)
axis.fill_between(
    x_array[900:],
    np.interp(x_array, range(24), solstice_narrow_475.values)[900:],
    np.interp(x_array, range(24), solstice_control_475.values)[900:],
    alpha=0.3,
    color="C4",
    hatch="//",
    label="Lost PAR",
)

plt.legend(fontsize=7)
plt.xlabel("Time of day")
plt.ylabel("Global irradiance (diffuse + beam) / W/m$^2$")
axis.tick_params(axis="both", which="major", labelsize=7)

tick_positions, tick_labels = plt.xticks()
new_labels = [
    entry._text.split(" ")[1][:-3] for entry in (tick_labels[::4] + [tick_labels[-1]])
]
plt.xticks(tick_positions[::4] + [tick_positions[-1]], new_labels)

plt.legend(loc="upper left", fontsize=7)

plt.savefig(
    f"par_lsp_solstice_comparison__{INDEX}.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()

# Plot the irradiance, capped off by the PAR, as an average over the polytunnel
plt.figure(figsize=(171 * MM, 100 * MM))

solstice_control_mean = control_total.iloc[
    segments_to_solstice : segments_to_solstice + 24
].mean(axis=1)
solstice_wide_mean = wide_total.iloc[
    segments_to_solstice : segments_to_solstice + 24
].mean(axis=1)
solstice_narrow_mean = narrow_total.iloc[
    segments_to_solstice : segments_to_solstice + 24
].mean(axis=1)

sns.lineplot(solstice_control_mean, label="No PV mean", color="C0")
sns.lineplot(solstice_wide_mean, label='"Sparse" mean', color="C1")
sns.lineplot(solstice_narrow_mean, label='"Dense" mean', color="C2")


def _interp_control(x: list[float]) -> list[float]:
    return np.interp(x, solstice_control_475.index, solstice_control_475.values)


(axis := plt.gca()).axhline(strawberry_lsp, dashes=(4, 4), color="C0", label="LSP")

x_array = np.linspace(0, 23, 1440)
# Plot power generated even in the narrow sense
axis.fill_between(
    x_array,
    [0] * 1440,
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_narrow_mean.values)
    ],
    alpha=0.1,
    color="C2",
    hatch="//",
    label='"Dense" power',
)

# axis.fill_between(
#     x_array[:900],
#     np.interp(x_array, range(24), solstice_narrow_mean.values)[:900],
#     np.interp(x_array, range(24), solstice_control_mean.values)[:900],
#     alpha=0.7,
#     color="#ABABAB",
#     hatch="//",
#     label="PAR above LSP",
# )
axis.fill_between(
    x_array,
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_narrow_mean.values)
    ],
    [
        min(entry, strawberry_lsp)
        for entry in np.interp(x_array, range(24), solstice_control_mean.values)
    ],
    alpha=0.3,
    color="C4",
    hatch="//",
    label="Lost PAR",
)

sum(
    np.array(
        [
            min(entry, strawberry_lsp)
            for entry in np.interp(x_array, range(24), solstice_control_mean.values)
        ]
    )
    - np.array(
        [
            min(entry, strawberry_lsp)
            for entry in np.interp(x_array, range(24), solstice_narrow_mean.values)
        ]
    )
) / sum(
    np.array(
        [
            min(entry, strawberry_lsp)
            for entry in np.interp(x_array, range(24), solstice_control_mean.values)
        ]
    )
)

plt.legend(fontsize=7)
plt.xlabel("Time of day")
plt.ylabel("Mean global irradiance / W/m$^2$")
axis.tick_params(axis="both", which="major", labelsize=7)

tick_positions, tick_labels = plt.xticks()
new_labels = [
    entry._text.split(" ")[1][:-3] for entry in (tick_labels[::4] + [tick_labels[-1]])
]
plt.xticks(tick_positions[::4] + [tick_positions[-1]], new_labels)

plt.legend(loc="upper left", fontsize=7, ncols=1)

plt.ylim(None, 550)

plt.savefig(
    f"par_lsp_solstice_comparison_mean__{INDEX}.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()

# Plot variation across the polytunnel throughout the day
plt.figure(figsize=(171 * MM, 100 * MM))
solstice_narrow = narrow_total.iloc[segments_to_solstice : segments_to_solstice + 24]

sns.lineplot(
    (
        y_val := solstice_narrow.transpose()[:100]
        .transpose()
        .clip(upper=strawberry_lsp)
        .mean(axis=1)
    ),
    color="C0",
    label="Edge regions",
)
# yerr = solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).std(axis=1)
# plt.fill_between(
#     yerr.index,
#     y_val-yerr,
#     y_val+yerr,
#     alpha=0.1,
#     color="C0",
# )

ymin = (
    solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).min(axis=1)
)
ymax = (
    solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).max(axis=1)
)
plt.fill_between(
    y_val.index,
    ymin,
    ymax,
    alpha=0.05,
    color="C0",
)

sns.lineplot(
    (
        y_val := solstice_narrow.transpose()[100:200]
        .transpose()
        .clip(upper=strawberry_lsp)
        .mean(axis=1)
    ),
    color="C1",
    label="Centre-West",
)
# yerr = solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).std(axis=1)
# plt.fill_between(
#     yerr.index,
#     y_val-yerr,
#     y_val+yerr,
#     alpha=0.1,
#     color="C0",
# )

ymin = (
    solstice_narrow.transpose()[100:200]
    .transpose()
    .clip(upper=strawberry_lsp)
    .min(axis=1)
)
ymax = (
    solstice_narrow.transpose()[100:200]
    .transpose()
    .clip(upper=strawberry_lsp)
    .max(axis=1)
)
plt.fill_between(
    y_val.index,
    ymin,
    ymax,
    alpha=0.1,
    color="C1",
)


sns.lineplot(
    (
        y_val := solstice_narrow.transpose()[200:300]
        .transpose()
        .clip(upper=strawberry_lsp)
        .mean(axis=1)
    ),
    color="C4",
    label="Centre",
)
# yerr = solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).std(axis=1)
# plt.fill_between(
#     yerr.index,
#     y_val-yerr,
#     y_val+yerr,
#     alpha=0.1,
#     color="C0",
# )

ymin = (
    solstice_narrow.transpose()[200:300]
    .transpose()
    .clip(upper=strawberry_lsp)
    .min(axis=1)
)
ymax = (
    solstice_narrow.transpose()[200:300]
    .transpose()
    .clip(upper=strawberry_lsp)
    .max(axis=1)
)
plt.fill_between(
    y_val.index,
    ymin,
    ymax,
    alpha=0.1,
    color="C4",
)

sns.lineplot(
    (
        y_val := solstice_narrow.transpose()[300:400]
        .transpose()
        .clip(upper=strawberry_lsp)
        .mean(axis=1)
    ),
    color="C2",
    label="Centre-East",
)
# yerr = solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).std(axis=1)
# plt.fill_between(
#     yerr.index,
#     y_val-yerr,
#     y_val+yerr,
#     alpha=0.1,
#     color="C0",
# )

ymin = (
    solstice_narrow.transpose()[300:400]
    .transpose()
    .clip(upper=strawberry_lsp)
    .min(axis=1)
)
ymax = (
    solstice_narrow.transpose()[300:400]
    .transpose()
    .clip(upper=strawberry_lsp)
    .max(axis=1)
)
plt.fill_between(
    y_val.index,
    ymin,
    ymax,
    alpha=0.1,
    color="C2",
)

sns.lineplot(
    (
        y_val := solstice_narrow.transpose()[400:500]
        .transpose()
        .clip(upper=strawberry_lsp)
        .mean(axis=1)
    ),
    color="C0",
)
# yerr = solstice_narrow.transpose()[:100].transpose().clip(upper=strawberry_lsp).std(axis=1)
# plt.fill_between(
#     yerr.index,
#     y_val-yerr,
#     y_val+yerr,
#     alpha=0.1,
#     color="C0",
# )

ymin = (
    solstice_narrow.transpose()[400:500]
    .transpose()
    .clip(upper=strawberry_lsp)
    .min(axis=1)
)
ymax = (
    solstice_narrow.transpose()[400:500]
    .transpose()
    .clip(upper=strawberry_lsp)
    .max(axis=1)
)
plt.fill_between(
    y_val.index,
    ymin,
    ymax,
    alpha=0.05,
    color="C0",
)

plt.legend(fontsize=7)
plt.xlabel("Time of day")
plt.ylabel("Mean global irradiance / W/m$^2$")
(axis := plt.gca()).tick_params(axis="both", which="major", labelsize=7)

tick_positions, tick_labels = plt.xticks()
new_labels = [
    entry._text.split(" ")[1][:-3] for entry in (tick_labels[::4] + [tick_labels[-1]])
]
plt.xticks(tick_positions[::4] + [tick_positions[-1]], new_labels)

plt.savefig(
    f"par_lsp_east_west_variation__{INDEX}.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()


# solstice_control = control_total.iloc[segments_to_solstice:segments_to_solstice+24]

# for _, row in solstice_control.iterrows():
#     sns.heatmap(np.reshape(row, (10, 50)), cmap="viridis", vmax=800)
#     plt.show()

# for _, row in solstice_narrow.iterrows():
#     sns.heatmap(np.reshape(row, (10, 50)), cmap="viridis", vmax=800)
#     plt.show()


# df_long = frame.melt(id_vars=frame.index
#                   value_vars=["control", "wide", "normal"],
#                   var_name="series",
#                   value_name="value")


# plt.figure(figsize=(180 * MM, 120 * MM))
# sns.barplot(narrow_daily)
# plt.show()

# sns.set_palette(pi_pv_palette)

# plt.figure(figsize=(180 * MM, 120 * MM))
# sns.scatterplot(
#     pd.DataFrame(
#         np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#     ).mean(axis=1),
#     color="C1",
# )
# sns.scatterplot(
#     pd.DataFrame(np.reshape(wide_total.mean(axis=1).values, (int(428 / 2), 48))).mean(
#         axis=1
#     ),
#     color="C3",
# )
# sns.scatterplot(
#     pd.DataFrame(
#         np.reshape(control_total.mean(axis=1).values, (int(428 / 2), 48))
#     ).mean(axis=1),
#     color="C4",
# )
# plt.show()

# _data = pd.DataFrame(
#     {
#         "narrow": pd.DataFrame(
#             np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#         "wide": pd.DataFrame(
#             np.reshape(wide_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#         "control": pd.DataFrame(
#             np.reshape(control_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#     }
# )


# _data = pd.DataFrame(
#     {
#         "narrow": pd.DataFrame(
#             np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#         "wide": pd.DataFrame(
#             np.reshape(wide_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#         "control": pd.DataFrame(
#             np.reshape(control_total.mean(axis=1).values, (int(428 / 2), 48))
#         )
#         .mean(axis=1)
#         .groupby(
#             pd.DataFrame(
#                 np.reshape(narrow_total.mean(axis=1).values, (int(428 / 2), 48))
#             )
#             .mean(axis=1)
#             .index
#             // 30
#         )
#         .mean(),
#     }
# )

# plt.figure(figsize=(180 * MM, 120 * MM))
# sns.barplot(_data.transpose(), hue=_data.index, dodge=True)
# plt.show()
