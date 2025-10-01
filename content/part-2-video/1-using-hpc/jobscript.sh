#!/bin/bash
#BSUB -J DLCV-part-2-exercise-1
#BSUB -q c02516
#BSUB -gpu "num1:mode=exclusive_process"

#BSUB -N

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

# Max wall clock time is 12 hours
#BSUB -W 1:00 

#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

source .venv/bin/activate
python simple_script.py