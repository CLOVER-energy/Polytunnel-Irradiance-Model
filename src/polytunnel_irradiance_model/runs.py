run_string: str = (
    """python -m src.polytunnel_irradiance_model -pt circular_{POLYTUNNEL}_short -mres 10 -st 2024-{MONTH}-{DAY}T00:00:00Z -et 2024-{MONTH}-{DAY}T23:59:59Z -d 0.55 -vi 275 -wf corrected_renewables_ninja_weather.csv -wado -mtr 60 --latitude 51.249814 --longitude 0.347779 -vf elinor_sunlight_data_daylight_only.csv"""
)

runs: list[tuple[str, str, str]] = [
    ("16", "05", "control"),
    ("17", "05", "control"),
    ("22", "05", "wide"),
    ("23", "05", "wide"),
    ("24", "05", "wide"),
    ("05", "06", "narrow"),
    ("06", "06", "narrow"),
    ("12", "06", "narrow"),
    ("13", "06", "narrow"),
    ("19", "06", "control"),
    ("20", "06", "control"),
    ("26", "06", "wide"),
    ("27", "06", "wide"),
    ("04", "07", "wide"),
    ("05", "07", "wide"),
    ("10", "07", "narrow"),
    ("11", "07", "narrow"),
    ("24", "07", "control"),
    ("25", "07", "control"),
]

import subprocess
from rich.progress import track

for run in track(runs):
    day, month, polytunnel = run
    subprocess.run(
        run_string.format(MONTH=month, DAY=day, POLYTUNNEL=polytunnel).split(" ")
    )
