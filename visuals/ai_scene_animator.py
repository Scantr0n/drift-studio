"""Animates a single AI-generated illustration into a seamless video loop.
Composition stays completely locked (no zoom/pan — see render_frame
docstring for why); life comes entirely from small layered elements
animating within the fixed frame: rain, water motion, a blinking light,
steam, water shimmer.

Same seamless-loop principle as visuals/generator.py: every motion term
is an integer multiple of 2*pi*t, so frame(t=0) matches frame(t≈1) exactly
— verify this empirically after any change, don't just trust the math.

Region params (rain_region, water_motion_region, etc.) are (x0,y0,x1,y1)
as FRACTIONS of the output frame — always find these empirically by
rendering a test frame and checking placement, not by estimating from
the source image, since the crop offsets coordinates (see render_loop).
"""

import subprocess

import numpy as np
from PIL import Image, ImageDraw


def _safe_crop_box(src_w, src_h, target_aspect, zoom, pan_x_frac, pan_y_frac):
    if src_w / src_h > target_aspect:
        base_h = src_h
        base_w = src_h * target_aspect
    else:
        base_w = src_w
        base_h = src_w / target_aspect

    crop_w = base_w / zoom
    crop_h = base_h / zoom

    margin_x = (src_w - crop_w) / 2
    margin_y = (src_h - crop_h) / 2
    center_x = src_w / 2 + pan_x_frac * margin_x
    center_y = src_h / 2 + pan_y_frac * margin_y

    x0 = max(0, min(src_w - crop_w, center_x - crop_w / 2))
    y0 = max(0, min(src_h - crop_h, center_y - crop_h / 2))
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _region_px(region_frac, width, height):
    x0, y0, x1, y1 = region_frac
    return x0 * width, y0 * height, x1 * width, y1 * height


def _rain_overlay(img, t, width, height, region=None, n_streaks=70, speed_cycles=3,
                   angle_deg=10, seed=7, color=(220, 232, 248)):
    """Softer than the first version — Jack called the original 'a layer
    slapped on top that looks terrible'. Reduced density, varied per-streak
    opacity so it reads as atmospheric rather than a uniform mechanical
    grid, and can now be confined to a region (e.g. only outdoor areas,
    never over interior furniture where rain makes no sense)."""
    rng = np.random.default_rng(seed)
    if region:
        x0, y0, x1, y1 = _region_px(region, width, height)
    else:
        x0, y0, x1, y1 = 0, 0, width, height
    region_h = y1 - y0

    base_x = rng.uniform(x0, x1, n_streaks)
    base_y = rng.uniform(y0, y1, n_streaks)
    lengths = rng.uniform(10, 22, n_streaks)
    alphas = rng.uniform(20, 55, n_streaks)  # varied, and much lower than the original flat 80

    angle_rad = np.radians(angle_deg)
    dx_per_dy = np.tan(angle_rad)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(n_streaks):
        y = y0 + (base_y[i] - y0 + t * speed_cycles * region_h) % region_h
        x = base_x[i]
        y2 = y + lengths[i]
        x2 = x + lengths[i] * dx_per_dy
        draw.line([(x, y), (x2, y2)], fill=(*color, int(alphas[i])), width=1)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _glass_streak_overlay(img, t, region, width, height, n_streaks=22, cycles=2,
                           seed=17, color=(200, 218, 240, 130)):
    """Rain running down the INSIDE of a window pane, confined to the
    window region only — droplets trail downward with a slight wobble,
    fading behind them, instead of falling straight through open air.
    For an interior-view scene, this replaces open-air rain entirely."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = _region_px(region, width, height)
    region_h = y1 - y0

    base_x = rng.uniform(x0, x1, n_streaks)
    phase = rng.random(n_streaks)
    trail_len = rng.uniform(30, 70, n_streaks)
    wobble_amp = rng.uniform(2, 6, n_streaks)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(n_streaks):
        head_y = y0 + ((t * cycles + phase[i]) % 1.0) * region_h
        wobble = wobble_amp[i] * np.sin(2 * np.pi * (cycles * 3 * t + phase[i] * 5))
        head_x = base_x[i] + wobble
        n_segments = 8
        for s in range(n_segments):
            seg_frac = s / n_segments
            seg_y = head_y - seg_frac * trail_len[i]
            if seg_y < y0:
                continue
            seg_wobble = wobble_amp[i] * np.sin(2 * np.pi * (cycles * 3 * (t - seg_frac * 0.02) + phase[i] * 5))
            seg_x = base_x[i] + seg_wobble
            alpha = int(color[3] * (1 - seg_frac))
            r = 1.6 * (1 - seg_frac * 0.5)
            draw.ellipse([seg_x - r, seg_y - r, seg_x + r, seg_y + r], fill=(*color[:3], alpha))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _water_motion_overlay(img, t, region, width, height, n_streams=18, cycles=5,
                           seed=19, color=(235, 245, 250)):
    """Fast pale vertical streaks confined to a narrow region — for a
    waterfall/rushing water, not general rain. Faster and whiter than
    rain so it reads as churning water, not falling droplets."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = _region_px(region, width, height)
    region_h = y1 - y0

    base_x = rng.uniform(x0, x1, n_streams)
    base_y = rng.uniform(y0, y1, n_streams)
    lengths = rng.uniform(8, 20, n_streams)
    alphas = rng.uniform(40, 90, n_streams)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(n_streams):
        y = y0 + (base_y[i] - y0 + t * cycles * region_h) % region_h
        x = base_x[i]
        draw.line([(x, y), (x, y + lengths[i])], fill=(*color, int(alphas[i])), width=1)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _steam_overlay(img, t, pos_frac, width, height, n_wisps=3, cycles=1,
                    seed=23, rise_height=70, color=(235, 235, 235)):
    """A few soft wavering wisps rising from a fixed point (e.g. a coffee
    cup) and fading out — wraps via the same modulo technique as falling
    rain, just inverted (rising instead of falling)."""
    rng = np.random.default_rng(seed)
    x, y = pos_frac[0] * width, pos_frac[1] * height
    phase = rng.random(n_wisps)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(n_wisps):
        life = (t * cycles + phase[i]) % 1.0
        wisp_y = y - life * rise_height
        wobble = 6 * np.sin(2 * np.pi * (cycles * 2 * t + phase[i] * 3)) * life
        wisp_x = x + wobble
        alpha = int(90 * np.sin(np.pi * life))  # fades in then out over its rise
        r = 3 + 5 * life
        draw.ellipse([wisp_x - r, wisp_y - r, wisp_x + r, wisp_y + r], fill=(*color, max(0, alpha)))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _blinking_light(img, t, width, height, pos_frac, radius=6, color=(255, 60, 50), blink_hz=1.0):
    x, y = pos_frac[0] * width, pos_frac[1] * height
    phase = (t * blink_hz) % 1.0
    if phase >= 0.5:
        return img
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r, alpha in [(radius * 2.5, 60), (radius * 1.4, 130), (radius, 220)]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*color, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _water_shimmer(img, t, region_frac, width, height, n_sparkles=25,
                    cycles=4, seed=11, color=(255, 250, 220)):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = _region_px(region_frac, width, height)
    base_x = rng.uniform(x0, x1, n_sparkles)
    base_y = rng.uniform(y0, y1, n_sparkles)
    phase_offset = rng.random(n_sparkles)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(n_sparkles):
        b = 0.5 + 0.5 * np.sin(2 * np.pi * (cycles * t + phase_offset[i]))
        if b < 0.6:
            continue
        alpha = int((b - 0.6) / 0.4 * 200)
        r = 2.5
        draw.ellipse([base_x[i] - r, base_y[i] - r, base_x[i] + r, base_y[i] + r],
                     fill=(*color, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def render_frame(source_image, t, width, height, static_crop, effects):
    """effects: list of (fn_name, kwargs) applied in order. Keeping this
    as an ordered list (rather than a fixed set of named params) since
    different scenes need genuinely different combinations, not just
    on/off toggles of the same fixed set."""
    cropped = source_image.crop(static_crop).resize((width, height), Image.LANCZOS)

    fn_map = {
        "rain": _rain_overlay,
        "glass_streak": _glass_streak_overlay,
        "water_motion": _water_motion_overlay,
        "steam": _steam_overlay,
        "blinking_light": _blinking_light,
        "water_shimmer": _water_shimmer,
    }
    for name, kwargs in effects:
        cropped = fn_map[name](cropped, t, width=width, height=height, **kwargs)

    return cropped


def render_loop(image_path, duration_sec, fps, width, height, out_path, effects):
    source_image = Image.open(image_path).convert("RGB")
    src_w, src_h = source_image.size
    static_crop = _safe_crop_box(src_w, src_h, width / height, zoom=1.0, pan_x_frac=0, pan_y_frac=0)
    n_frames = int(duration_sec * fps)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_idx in range(n_frames):
        t = frame_idx / n_frames
        frame = render_frame(source_image, t, width, height, static_crop, effects)
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='ignore')}")
    return out_path
