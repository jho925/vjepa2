#!/usr/bin/env python3
# Run V-JEPA 2 1B @ 384 on frames from a directory.
# Expected frame names: a.000000.png, a.000001.png, a.000002.png, ... (sorted by numeric suffix).
# Usage: from vjepa2 root: python run_inference_frames.py [--frames_dir inference_data/frames/sample_1] [--output results/features.pt]

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

# Default checkpoint: shared lab path (download vitg-384.pt there once)
DEFAULT_CHECKPOINT = "/n/netscratch/koumoutsakos_lab/Lab/shared/vitg-384.pt"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights loaded from %s: %s" % (pretrained_weights, msg))


def _frame_sort_key(path):
    """Sort by numeric suffix (e.g. a.000001.png -> 1) for temporal order."""
    stem = path.stem  # e.g. "a.000001"
    parts = stem.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


def load_frames_from_dir(frames_dir):
    paths = []
    for ext in FRAME_EXTENSIONS:
        paths.extend(frames_dir.glob("*" + ext))
    paths = sorted(paths, key=_frame_sort_key)
    if not paths:
        raise FileNotFoundError(
            "No image files in %s (extensions: %s)" % (frames_dir, ", ".join(FRAME_EXTENSIONS))
        )
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))
    return frames


def run_inference(frames_dir, output_path, device, num_frames_max=None, checkpoint_path=None):
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        sys.exit("Frames directory not found: %s" % frames_dir)
    print("Loading frames from %s ..." % frames_dir)
    frames = load_frames_from_dir(frames_dir)
    total_available = len(frames)
    if num_frames_max is not None:
        frames = frames[:num_frames_max]
        print("Using first %d of %d frames." % (len(frames), total_available))
    num_frames = len(frames)
    h, w = frames[0].shape[:2]
    print("Loaded %d frames. Initial frame size (H, W): %d x %d" % (num_frames, h, w))
    if num_frames == 0:
        sys.exit("No frames to process.")
    img_size = 384
    transform = build_pt_video_transform(img_size)
    video_tensor = transform(frames)
    x = video_tensor.unsqueeze(0).to(device)
    print("Input shape: %s (B, C, T, H, W)" % (x.shape,))
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path(DEFAULT_CHECKPOINT)
    if checkpoint_path.exists():
        print("Loading V-JEPA 2 1B @ 384 (encoder) from %s ..." % checkpoint_path)
        encoder = vit_giant_xformers_rope(img_size=(384, 384), num_frames=64)
        load_pretrained_vjepa_pt_weights(encoder, str(checkpoint_path))
    else:
        print("Loading V-JEPA 2 1B @ 384 (encoder) from Torch Hub (no checkpoint at %s) ..." % checkpoint_path)
        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_vit_giant_384",
            pretrained=True,
            trust_repo=True,
        )
    encoder = encoder.to(device).eval()
    print("Running inference ...")
    with torch.inference_mode():
        features = encoder(x)
    print("Output features shape: %s" % (features.shape,))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.cpu(), "num_frames": num_frames}, out)
    print("Saved features to %s" % out)
    return features


def main():
    parser = argparse.ArgumentParser(description="Run V-JEPA 2 1B @ 384 on frames from a directory.")
    parser.add_argument("--frames_dir", default="inference_data/frames/sample_1", help="Directory of frame images (e.g. inference_data/frames/sample_2)")
    parser.add_argument("--num_frames", type=int, default=None, help="Use only the first N frames (default: use all)")
    parser.add_argument("--checkpoint", default=None, help="Encoder checkpoint .pt (default: %s)" % DEFAULT_CHECKPOINT)
    parser.add_argument("-o", "--output", default=None, help="Path to save features (default: results/<sample_name>/features.pt)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.output is None:
        sample_name = Path(args.frames_dir).name
        args.output = "results/%s/features.pt" % sample_name
    run_inference(
        args.frames_dir,
        args.output,
        device,
        num_frames_max=args.num_frames,
        checkpoint_path=args.checkpoint or DEFAULT_CHECKPOINT,
    )


if __name__ == "__main__":
    main()
