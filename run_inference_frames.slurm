#!/bin/bash
# Usage: sbatch run_inference_frames.slurm [sample_name] [num_frames]
# Example: sbatch run_inference_frames.slurm sample_1
# Example: sbatch run_inference_frames.slurm sample_2 32
# Logs go to results/<sample_name>/slurm-<jobid>.{out,err}
#SBATCH --job-name=vjepa2-infer
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64000
#SBATCH -t 0-01:00
#SBATCH --account=koumoutsakos_lab

SAMPLE="${1:-sample_1}"
NUM_FRAMES="${2:-}"

# See https://docs.rc.fas.harvard.edu/kb/running-jobs/
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vjepa

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "results/$SAMPLE"
exec 1>"results/$SAMPLE/slurm-${SLURM_JOB_ID}.out" 2>"results/$SAMPLE/slurm-${SLURM_JOB_ID}.err"

CHECKPOINT="/n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt"
if [ -n "$NUM_FRAMES" ]; then
  python run_inference_frames.py --frames_dir "inference_data/frames/$SAMPLE" --checkpoint "$CHECKPOINT" --num_frames "$NUM_FRAMES"
else
  python run_inference_frames.py --frames_dir "inference_data/frames/$SAMPLE" --checkpoint "$CHECKPOINT"
fi
