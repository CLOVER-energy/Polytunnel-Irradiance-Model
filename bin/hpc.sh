########################################################################################
# hpc.sh - Runs the Polytunnel-Irradiance model as an HPC job.                         #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2026                                                      #
# Date created: 03/06/2026                                                             #
# License: Open source                                                                 #
# Most recent update: 03/06/2026                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=128Gb
#PBS -N ppv-pir

# Depending on the environmental variable, run the appropriate HPC job.
module load anaconda3/personal
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py310

# Change to the submission directory
cd $PBS_O_WORKDIR

# Determine the scenario to run
python -m src.polytunnel_irradiance_model -pt circular_narrow_short_mariano \
    -mres 10 -st 2024-06-01T00:00:00Z -et 2024-06-30T23:59:59Z \
    -d 0.55 -vi 275 -wf ninja_16_25_kent.csv -wado -mtr 60 \
    -lat 51.249814 -lon 0.347779 -sp -hwf cosmos_hadlow_1624.csv 
