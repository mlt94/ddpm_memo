#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J mlut
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 02:00
# request 5GB of system-memory and 32GB of GPU-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
# -- end of LSF options --


# activate the virtual environment
source 
python /detect_mem.py --run_name detect_memorization_validation --end 200 --model_id "" 
