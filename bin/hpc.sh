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
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=32Gb
#PBS -N ppv-pir
#PBS -J 1-3137

# Depending on the environmental variable, run the appropriate HPC job.
module load anaconda3/personal
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate ppv

# Change to the submission directory
cd $PBS_O_WORKDIR

# Determine the scenario to run
python -m src.polytunnel_irradiance_model.hpc -pt circular_control
