#!/bin/bash
# Usage: sbatch run_inference_frames.slurm <frames_dir> [output_name] [num_frames]
# Output: results/<output_name>/features.pt (default output_name = last dir of frames_dir)
# Example: sbatch run_inference_frames.slurm /path/to/periodicHill/frames periodicHill
# Example: sbatch run_inference_frames.slurm inference_data/frames/sample_1 sample_1 32
#SBATCH --job-name=vjepa2-infer
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64000
#SBATCH -t 0-01:00
#SBATCH --account=koumoutsakos_lab

FRAMES_DIR="${1:?Usage: sbatch run_inference_frames.slurm <frames_dir> [output_name] [num_frames]}"
OUTNAME="${2:-$(basename "${FRAMES_DIR%/}")}"
NUM_FRAMES="${3:-}"
CHECKPOINT="/n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vjepa

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "results/$OUTNAME"
exec 1>"results/$OUTNAME/slurm-${SLURM_JOB_ID}.out" 2>"results/$OUTNAME/slurm-${SLURM_JOB_ID}.err"

if [ -n "$NUM_FRAMES" ]; then
  python run_inference_frames.py --frames_dir "$FRAMES_DIR" --checkpoint "$CHECKPOINT" -o "results/$OUTNAME/features.pt" --num_frames "$NUM_FRAMES"
else
  python run_inference_frames.py --frames_dir "$FRAMES_DIR" --checkpoint "$CHECKPOINT" -o "results/$OUTNAME/features.pt"
fi
