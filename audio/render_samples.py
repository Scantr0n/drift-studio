"""Renders a short sample of each noise type for a quick listen/QA check."""

import os
from generators import NOISE_TYPES, render_to_wav

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in NOISE_TYPES:
        path = os.path.join(OUT_DIR, f"sample_{name}_noise.wav")
        render_to_wav(name, duration_sec=45, out_path=path, seed=42)
        print(f"wrote {path}")
