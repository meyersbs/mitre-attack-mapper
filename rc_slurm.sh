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
#SBATCH -J mitre_attack

# Files to save STDOUT and STDERR to.
# Format: -o <stdout_filename>.output
# Format: -e <stderr_filename>.outout
#SBATCH -o mitre_attack.output
#SBATCH -e mitre_attack.output

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
#SBATCH -p tier3 -A csec -n 1 -c 16

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
time ./main.py info >> info.out
time ./main.py train nb tactics >> nb_tactics.out
time ./main.py train nb techniques >> nb_techniques.out
time ./main.py train lsvc tactics >> lsvc_tactics.out
time ./main.py train lsvc techniques >> lsvc_techniques.out
