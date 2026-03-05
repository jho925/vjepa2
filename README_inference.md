# Running V-JEPA 2 inference on frames

This guide explains how to run **`run_inference_frames.py`** (V-JEPA 2 1B @ 384 encoder) on frame images, either locally or via Slurm.

---

## Prerequisites

- **Conda env:** Activate the `vjepa` environment (or whichever env has the vjepa2 dependencies).
- **Frames:** Put frame images in a folder under `inference_data/frames/<sample_name>/`, e.g.:
  - `inference_data/frames/sample_1/`
  - `inference_data/frames/sample_2/`
- **Frame names:** Use the pattern `a.000000.png`, `a.000001.png`, … (sorted by numeric suffix). Supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- **Model:** The script loads the 1B @ 384 encoder from **`/n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt`** if that file exists; otherwise it falls back to **Torch Hub**. To use the shared model once, download it there:
  ```bash
  mkdir -p /n/netscratch/koumoutsakos_lab/Lab/shared
  wget -O /n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt
  ```

---

## 1. Run the Python script directly (local or login node)

From the **vjepa2 repo root**:

```bash
conda activate vjepa
cd /path/to/vjepa2

# Default: uses inference_data/frames/sample_1, writes results/sample_1/features.pt
python run_inference_frames.py

# Specific sample (output goes to results/sample_2/features.pt)
python run_inference_frames.py --frames_dir inference_data/frames/sample_2

# Use only the first 32 frames from the sample
python run_inference_frames.py --frames_dir inference_data/frames/sample_2 --num_frames 32

# Custom output path
python run_inference_frames.py --frames_dir inference_data/frames/sample_2 -o results/sample_2/features.pt

# CPU only
python run_inference_frames.py --frames_dir inference_data/frames/sample_1 --device cpu
```

**Arguments:**

| Argument        | Default                       | Description                                      |
|----------------|-------------------------------|--------------------------------------------------|
| `--frames_dir` | `inference_data/frames/sample_1` | Directory containing frame images.             |
| `--num_frames` | (all)                         | Use only the first N frames from the directory. |
| `-o` / `--output` | (auto)                    | Output path. If omitted: `results/<sample_name>/features.pt` (sample_name = last part of `--frames_dir`). |
| `--device`     | auto (cuda if available)      | Device: `cuda` or `cpu`.                         |

**Output:** A `.pt` file containing `{"features": tensor, "num_frames": int}`.

---

## 2. Run via Slurm (GPU batch job)

Submit from the **vjepa2 repo root**. Logs and features go into **`results/<sample_name>/`**.

```bash
cd /path/to/vjepa2

# Run for sample_1 (default)
sbatch run_inference_frames.slurm

# Run for sample_2
sbatch run_inference_frames.slurm sample_2

# Run for sample_3, etc.
sbatch run_inference_frames.slurm sample_3

# Run for sample_2 using only the first 32 frames
sbatch run_inference_frames.slurm sample_2 32
```

**Important:** Pass the **sample name** as the first argument. That name must match a folder under `inference_data/frames/` (e.g. `sample_2` → `inference_data/frames/sample_2/`).

**Where output goes:**

| You run                         | Frames read from                    | Output and logs |
|---------------------------------|-------------------------------------|------------------|
| `sbatch run_inference_frames.slurm` | `inference_data/frames/sample_1/` | `results/sample_1/features.pt`, `results/sample_1/slurm-<jobid>.out`, `results/sample_1/slurm-<jobid>.err` |
| `sbatch run_inference_frames.slurm sample_2` | `inference_data/frames/sample_2/` | `results/sample_2/features.pt`, `results/sample_2/slurm-<jobid>.out`, `results/sample_2/slurm-<jobid>.err` |

**Slurm script defaults (edit `run_inference_frames.slurm` if needed):**

- Partition: `gpu`
- 1 GPU, 4 CPUs, 64 GB RAM, 1 hour
- Account: `koumoutsakos_lab`

**Useful commands:**

```bash
squeue -u $USER          # list your jobs
scancel <job_id>        # cancel one job
scancel -u $USER        # cancel all your jobs
```

---

## 3. Model and behavior

- **Model:** V-JEPA 2 1B @ 384 (ViT-giant, 384×384), encoder only.
- **Loaded via:** Local path `/n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt` if present; else Torch Hub.
- **Input:** Frames are resized, center-cropped to 384×384, and ImageNet-normalized. Variable number of frames is supported (e.g. 51).
- **Output:** Encoder features (no prediction of future frames; feature extraction only).
