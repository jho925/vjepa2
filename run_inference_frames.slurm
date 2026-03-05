#!/bin/bash
#SBATCH --job-name=vjepa2-infer
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64000
#SBATCH -t 0-01:00
#SBATCH --account=koumoutsakos_lab

# See https://docs.rc.fas.harvard.edu/kb/running-jobs/
# Activate conda (adjust path if your conda is elsewhere)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vjepa

# Run from directory where sbatch was submitted (must submit from vjepa2 repo root)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results

python run_inference_frames.py \
  --frames_dir inference_data/frames \
  --output results/features.pt
