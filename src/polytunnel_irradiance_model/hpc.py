########################################################################################
# hpc.py --- Entry point for running the irradiance model from the HPC.                #
#                                                                                      #
# Author(s): Benedict Winchester                                                       #
# Date created: Summer 2026                                                            #
#                                                                                      #
########################################################################################

"""
Polytunnel Irradiance Model: `hpc.py`

Entry point for the model when running on the hpc.

"""

import datetime
import os
import sys

from typing import Any

from src.polytunnel_irradiance_model.__main__ import main as ppv_model_main

# BASE_ARGUMENTS:
#   The base arguments to pass through to the main module..
BASE_ARGUMENTS: str = (
    "-pt circular_narrow_short_mariano -mres 10 -st {start_time} -et {end_time} "
    "-d 0.55 -vi 275 -wf ninja_16_25_kent.csv -wado -mtr 60 -lat 51.249814 "
    "-lon 0.347779 -sp -hwf cosmos_hadlow_1624.csv"
)

# HPC_JOB_NUMBER_VAR:
#   The name of the environment variable which stores the HPC job number.
HPC_JOB_NUMBER_VAR: str = "PBS_ARRAY_INDEX"

# START_DATE:
#   The hard-coded start date.
START_DATE: datetime.datetime = datetime.datetime(2016, 10, 28)


def main(args: list[Any]) -> None:
    """
    Main function for calling the module when run on the HPC.

    """

    # Utilise the environment variable to determine the number of the job.
    run_number: int | str = os.environ.get(HPC_JOB_NUMBER_VAR, None)

    if run_number is None:
        raise Exception(
            "No run number environment variable set: script must be run as part of an "
            "array job on the HPC."
        )

    try:
        run_number = int(run_number)
    except ValueError:
        raise Exception("Environment variable is not of the correct type.")

    # Determine the date based on the variable provided
    _timedelta = datetime.timedelta(days=run_number - 1)

    # Create the command string.
    updated_arguments = BASE_ARGUMENTS.format(
        start_time=(START_DATE + _timedelta).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_time=(
            START_DATE
            + _timedelta
            + datetime.timedelta(hours=23, minutes=59, seconds=59)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    ppv_model_main(updated_arguments.split(" "))


if __name__ == "__main__":
    main(sys.argv[1:])
