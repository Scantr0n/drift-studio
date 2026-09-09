"""Animated background generator for Drift Noise / Drift Sounds videos.

Renders a short seamlessly-looping clip once (drifting gradient blobs +
a particle layer), which the video pipeline then loops via ffmpeg to
fill an hour or ten hours — we never render more than one loop's worth
of unique frames.

Looping works by parameterizing every motion as an integer multiple of
2*pi*t where t in [0, 1). cos/sin of an integer multiple of 2*pi*t is
identical at t=0 and t=1, so frame 0 and the frame just past the last
one match exactly — no visible seam.
"""

import subprocess
import numpy as np

PRESETS = {
    # Drift Noise — abstract palettes, particles drift gently in place
    "white": {"palette": [(240, 240, 245), (210, 215, 230), (225, 225, 230)], "particle_color": (255, 255, 255), "particle_style": "drift"},
    "pink": {"palette": [(255, 200, 210), (240, 170, 190), (255, 220, 215)], "particle_color": (255, 235, 240), "particle_style": "drift"},
    "brown": {"palette": [(120, 80, 50), (90, 60, 40), (150, 100, 60)], "particle_color": (200, 160, 110), "particle_style": "drift", "particle_speed": 0.6},
    "red": {"palette": [(140, 30, 40), (90, 20, 30), (160, 50, 50)], "particle_color": (220, 120, 110), "particle_style": "drift"},
    "green": {"palette": [(50, 110, 80), (30, 80, 60), (70, 130, 90)], "particle_color": (170, 220, 170), "particle_style": "drift"},

    # Drift Sounds — scene-matched palettes and particle motion
    "rain": {"palette": [(60, 70, 90), (40, 50, 70), (70, 85, 100)], "particle_color": (200, 220, 235), "particle_style": "fall", "particle_speed": 2.5, "n_particles": 70},
    "wind": {"palette": [(150, 180, 190), (120, 150, 165), (170, 195, 200)], "particle_color": (230, 240, 240), "particle_style": "drift", "particle_speed": 1.5},
    "thunder": {"palette": [(45, 40, 55), (25, 22, 35), (55, 48, 65)], "particle_color": (140, 140, 160), "particle_style": "drift", "n_particles": 15},
    "birds": {"palette": [(200, 175, 100), (140, 170, 90), (220, 200, 140)], "particle_color": (255, 235, 180), "particle_style": "rise", "particle_speed": 0.8},
    "cafe": {"palette": [(120, 85, 60), (150, 110, 75), (100, 70, 50)], "particle_color": (255, 210, 160), "particle_style": "rise", "particle_speed": 0.4, "n_particles": 25},
    "city": {"palette": [(20, 20, 40), (35, 30, 55), (15, 15, 30)], "particle_color": (255, 210, 120), "particle_style": "drift", "particle_speed": 0.3, "n_particles": 50},
}


def render_frame(t, width, height, palette, n_particles=40, particle_color=None,
                  particle_speed=1.0, particle_style="drift", seed=42):
    xs = np.arange(width)
    ys = np.arange(height)
    X, Y = np.meshgrid(xs, ys)

    img = np.zeros((height, width, 3), dtype=np.float64)
    n_blobs = len(palette)
    blob_radius = 0.48 * min(width, height)
    for i, color in enumerate(palette):
        freq = i + 1  # integer multiple -> seamless loop at t=0/1
        phase = i / n_blobs
        cx = width * (0.5 + 0.35 * np.cos(2 * np.pi * (freq * t + phase)))
        cy = height * (0.5 + 0.35 * np.sin(2 * np.pi * (freq * t + phase + 0.2)))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        weight = np.exp(-dist2 / (2 * blob_radius ** 2))
        for c in range(3):
            img[:, :, c] += weight * color[c]

    max_val = img.max()
    if max_val > 0:
        img = img / max_val * 235
    img = np.clip(img, 0, 255)

    if n_particles > 0:
        rng = np.random.default_rng(seed)
        base_x = rng.random(n_particles) * width
        base_y = rng.random(n_particles) * height
        pcolor = particle_color or (255, 255, 255)
        p_radius = max(2, min(width, height) * 0.004)
        # particle_speed must resolve to a whole number of cycles per loop,
        # same reason as blob freq above — otherwise the wraparound/motion
        # lands at a different spot at t=1 than where it started at t=0.
        speed_n = max(1, round(particle_speed))

        # Gaussian glow is negligible past ~6 sigma, so computing it across
        # the full width*height frame for every particle (as before) does
        # ~800x more work than needed at 1080p — confirmed by benchmark
        # (10s of footage took over 2 minutes). Only touch a small local
        # box around each particle instead; wraparound is handled by
        # taking the box's coordinates mod width/height before indexing.
        box = max(1, int(p_radius * 6))
        local = np.arange(-box, box)
        DX, DY = np.meshgrid(local, local)
        glow_shape = np.exp(-(DX ** 2 + DY ** 2) / (2 * p_radius ** 2)) * 180

        for i in range(n_particles):
            px0, py0 = base_x[i], base_y[i]
            if particle_style == "fall":
                py = (py0 + t * speed_n * height) % height
                px = px0
            elif particle_style == "rise":
                py = (py0 - t * speed_n * height) % height
                px = (px0 + 0.02 * width * np.sin(2 * np.pi * (t + py0 / height))) % width
            else:  # drift
                px = (px0 + 0.05 * width * np.cos(2 * np.pi * (speed_n * t + py0 / height))) % width
                py = (py0 + 0.05 * height * np.sin(2 * np.pi * (speed_n * t + px0 / width))) % height

            x_idx = (np.arange(int(px) - box, int(px) + box)) % width
            y_idx = (np.arange(int(py) - box, int(py) + box)) % height
            for c in range(3):
                region = img[y_idx[:, None], x_idx[None, :], c]
                img[y_idx[:, None], x_idx[None, :], c] = np.minimum(255, region + glow_shape * (pcolor[c] / 255))

    return img.astype(np.uint8)


def render_loop(preset_name, duration_sec, fps, width, height, out_path):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(PRESETS)}")
    cfg = PRESETS[preset_name]
    n_frames = int(duration_sec * fps)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_idx in range(n_frames):
        t = frame_idx / n_frames
        frame = render_frame(
            t, width, height,
            palette=cfg["palette"],
            n_particles=cfg.get("n_particles", 40),
            particle_color=cfg.get("particle_color"),
            particle_speed=cfg.get("particle_speed", 1.0),
            particle_style=cfg.get("particle_style", "drift"),
        )
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='ignore')}")
    return out_path
