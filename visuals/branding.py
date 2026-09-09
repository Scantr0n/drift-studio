"""Channel art (banner + profile picture) generated from the same
gradient/particle system as the video backgrounds, so channel branding
visually matches the content instead of being a separate design system.

A channel spans several sub-presets (Drift Noise = white/pink/brown/red/
green; Drift Sounds = rain/wind/thunder/birds/cafe/city), so branding
pulls one representative color from each sub-preset into a single
blended palette, rather than picking just one video's look to represent
the whole channel.

YouTube banner spec: 2560x1440 upload, but only the centered 1546x423
"safe area" is guaranteed visible across TV/desktop/mobile — text and
logos should stay centered so they land inside that zone regardless of
device. Profile pictures render as a circle (800x800 recommended), so
keep content centered there too since corners get cropped.
"""

import os

from PIL import Image, ImageDraw, ImageFont

from visuals.generator import render_frame, PRESETS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "branding")

CHANNELS = {
    "drift_noise": {"sub_presets": ["white", "pink", "brown", "red", "green"], "title": "DRIFT NOISE",
                    "flagship": "brown", "monogram": "N"},
    "drift_sounds": {"sub_presets": ["rain", "wind", "thunder", "birds", "cafe", "city"], "title": "DRIFT SOUNDS",
                      "flagship": "rain", "monogram": "S"},
}


def _font(size):
    for candidate in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _channel_palette(sub_presets):
    return [PRESETS[name]["palette"][0] for name in sub_presets]


def _draw_outlined_text(draw, xy, text, font):
    x, y = xy
    for ox, oy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
        draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def make_banner(channel_key, out_path, width=2560, height=1440):
    cfg = CHANNELS[channel_key]
    palette = _channel_palette(cfg["sub_presets"])
    frame = render_frame(0.3, width, height, palette=palette, n_particles=70,
                          particle_color=(255, 255, 255), particle_style="drift")
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font = _font(int(height * 0.085))
    text = cfg["title"]
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # centered in the full canvas == centered in the safe area, since the
    # safe area is itself centered
    _draw_outlined_text(draw, ((width - tw) / 2, (height - th) / 2 - bbox[1]), text, font)

    img.save(out_path)
    return out_path


def make_pfp(channel_key, out_path, size=800):
    """Avatars display as small as ~40px in comments/subscription lists,
    where the banner's multi-color blend + fine particles just turn into
    an indistinct blur (verified by downscaling and looking). So this
    uses a different, size-appropriate design: one flagship color per
    channel (not the full blend) as a simple 2-tone radial gradient,
    plus a single bold letter — shapes and high contrast read at a
    glance; fine detail doesn't."""
    cfg = CHANNELS[channel_key]
    flagship = PRESETS[cfg["flagship"]]["palette"][0]
    dark = tuple(int(c * 0.35) for c in flagship)
    frame = render_frame(0.0, size, size, palette=[flagship, dark], n_particles=0)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font = _font(int(size * 0.5))
    letter = cfg["monogram"]
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]), letter, font=font, fill=(255, 255, 255))

    img.save(out_path)
    return out_path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in CHANNELS:
        make_banner(key, os.path.join(OUT_DIR, f"{key}_banner.png"))
        make_pfp(key, os.path.join(OUT_DIR, f"{key}_pfp.png"))
        print(f"generated banner + pfp for {key}")
