#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a Serial Multi-Process job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Job name.
# Format: -J <job_name>
#SBATCH -J mitre_attack8

# Files to save STDOUT and STDERR to.
# Format: -o <stdout_filename>.output
# Format: -e <stderr_filename>.outout
#SBATCH -o mitre_attack8.output
#SBATCH -e mitre_attack8.output

# Email to send notifications to.
# Format: --mail-user <email@domain.com>
# Note: To disable emails, place two pound signs before the command.
#SBATCH --mail-user bsm9339@rit.edu

# Nptifications to send.
# Format: --mail-type=<BEGIN, END, FAIL, ALL>
#SBATCH --mail-type=ALL

# MAX runtime.
# Format: -t <days>-<hours>:<minutes>:<seconds>
#SBATCH -t 1-0:00:0

# Accouting info and required CPUs.
# Format: -p <partition> -A <project> -n <num_tasks> -c <num_cpus>
#SBATCH -p tier3 -A csec -n 1 -c 24

# Memory requirements.
# Format: --mem=<number><K,M,G,T>
# Note: Defaults to MB.
#SBATCH --mem=50G

#
# Your job script goes below this line.  
#

# Load python and required libraries
spack unload python
spack load py-scikit-learn@0.22 arch=linux-rhel7-x86_64
spack load py-setuptools@41.4.0 arch=linux-rhel7-x86_64

# Run the code

# TRIAL 01
#time ./main.py info ./data/CPTC2018.csv --target=tactics --ignore_singles=False >> trial01_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --trial_prefix=trial01 --ignore_singles=False >> trial01_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --trial_prefix=trial01 --ignore_singles=False >> trial01_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --trial_prefix=trial01 --ignore_singles=False >> trial01_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --trial_prefix=trial01 --ignore_singles=False >> trial01_lsvc_techniques.out

# TRIAL 02
#time ./main.py info ./data/CPTC2018.csv --target=tactics >> trial02_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=True --append_hosts=True --trial_prefix=trial02 --ignore_singles=False >> trial02_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=True --append_hosts=True --trial_prefix=trial02 --ignore_singles=False >> trial02_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=True --append_hosts=True --trial_prefix=trial02 --ignore_singles=False >> trial02_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=True --append_hosts=True --trial_prefix=trial02 --ignore_singles=False >> trial02_lsvc_techniques.out

# TRIAL 03
#time ./main.py info ./data/CPTC2018.csv --target=tactics --ignore_singles=False >> trial03_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=True --append_hosts=False --trial_prefix=trial03 --ignore_singles=False >> trial03_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=True --append_hosts=False --trial_prefix=trial03 --ignore_singles=False >> trial03_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=True --append_hosts=False --trial_prefix=trial03 --ignore_singles=False >> trial03_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=True --append_hosts=False --trial_prefix=trial03 --ignore_singles=False >> trial03_lsvc_techniques.out

# TRIAL 04
#time ./main.py info ./data/CPTC2018.csv --target=tactics --ignore_singles=False >> trial04_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=False --append_hosts=True --trial_prefix=trial04 --ignore_singles=False >> trial04_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=False --append_hosts=True --trial_prefix=trial04 --ignore_singles=False >> trial04_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=False --append_hosts=True --trial_prefix=trial04 --ignore_singles=False >> trial04_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=False --append_hosts=True --trial_prefix=trial04 --ignore_singles=False >> trial04_lsvc_techniques.out

# TRIAL 05
#time ./main.py info ./data/CPTC2018.csv --target=tactics --ignore_singles=True >> trial05_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --trial_prefix=trial05 --ignore_singles=True >> trial05_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --trial_prefix=trial05 --ignore_singles=True >> trial05_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --trial_prefix=trial05 --ignore_singles=True >> trial05_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --trial_prefix=trial05 --ignore_singles=True >> trial05_lsvc_techniques.out

# TRIAL 06
#time ./main.py info ./data/CPTC2018.csv --target=tactics --ignore_singles=True >> trial06_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=True --append_hosts=True --trial_prefix=trial06 --ignore_singles=True >> trial06_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=True --append_hosts=True --trial_prefix=trial06 --ignore_singles=True >> trial06_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=True --append_hosts=True --trial_prefix=trial06 --ignore_singles=True >> trial06_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=True --append_hosts=True --trial_prefix=trial06 --ignore_singles=True >> trial06_lsvc_techniques.out

# TRIAL 07
#time ./main.py info ./data/CPTC2018.csv --target=tactics >> trial07_info.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=True --append_hosts=False --trial_prefix=trial07 --ignore_singles=True >> trial07_nb_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=True --append_hosts=False --trial_prefix=trial07 --ignore_singles=True >> trial07_nb_techniques.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=True --append_hosts=False --trial_prefix=trial07 --ignore_singles=True >> trial07_lsvc_tactics.out
#time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=True --append_hosts=False --trial_prefix=trial07 --ignore_singles=True >> trial07_lsvc_techniques.out

# TRIAL 08
time ./main.py info ./data/CPTC2018.csv --target=tactics >> trial08_info.out
time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=tactics --append_states=False --append_hosts=True --trial_prefix=trial08 --ignore_singles=True >> trial08_nb_tactics.out
time ./main.py test ./data/CPTC2018.csv --model_type=nb --target=techniques --append_states=False --append_hosts=True --trial_prefix=trial08 --ignore_singles=True >> trial08_nb_techniques.out
time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=tactics --append_states=False --append_hosts=True --trial_prefix=trial08 --ignore_singles=True >> trial08_lsvc_tactics.out
time ./main.py test ./data/CPTC2018.csv --model_type=lsvc --target=techniques --append_states=False --append_hosts=True --trial_prefix=trial08 --ignore_singles=True >> trial08_lsvc_techniques.out

