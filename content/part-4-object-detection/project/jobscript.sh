#!/bin/bash
#BSUB -J Fast-RCNN-ObjDet
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications
#BSUB -N

# CPU cores and memory
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"

# Max wall clock time (increased for training)
#BSUB -W 2:00

# Output files
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

# Print job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Start Time: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# --- FIX: ENSURE CONDA IS INITIALIZED AND ENVIRONMENT IS ACTIVATED ---

# Source the main bash configuration to ensure 'conda' command is available.
uv run main.py

# --- CLEANUP ---

# Capture exit code from the python script
EXIT_CODE=$?


# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Exit with the same code as the main script
exit $EXIT_CODE
