#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J genimage
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:05
# request 5GB of system-memory and 32GB of GPU-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[gpu16gb]"
### -- set the email address --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo /dtu/blackhole/14/207860/memo/training/martin/gen_images.out
#BSUB -eo /dtu/blackhole/14/207860/memo/training/martin/gen_images.err
# -- end of LSF options --


# activate the virtual environment
source /zhome/ca/2/153088/memorization/venv/bin/activate
python /zhome/ca/2/153088/memorization/diffusion_memorization/own_gen.py
#python /zhome/ca/2/153088/memorization/diffusion_memorization/own_gen_fake_validationprompts.py
