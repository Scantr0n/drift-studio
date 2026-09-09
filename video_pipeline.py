"""Combines generated audio with a visual loop into a final video, and
generates upload metadata.

Sound sources come in two flavors that need different long-form strategies:
  - procedural (noise colors, rain/wind/thunder): generate directly at the
    full target duration in memory-safe chunks (see audio/generators.py
    and audio/ambience.py render_long_form).
  - sourced (birds/cafe/city): fixed-length real recordings, so we loop
    them. A hard cut loop would click at the seam, so we crossfade the
    tail into the head first (make_loopable) to get a seamless loop unit,
    then repeat that unit to fill the target duration.
"""

import os
import subprocess
import tempfile
import wave

import numpy as np
from scipy.io import wavfile

from audio import generators, ambience

SAMPLE_RATE = 44100
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
SOURCED_DIR = os.path.join(AUDIO_DIR, "sourced")
VISUALS_DIR = os.path.join(os.path.dirname(__file__), "visuals")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SOURCED_FILES = {
    "birds": "birds_morning.mp3",
    "cafe": "cafe_restaurant_chatter.mp3",
    # city_night.mp3 and the original rain_ambience.mp3 both turned out to
    # have crickets/birds baked into the field recording, caught on
    # listen 2026-08-16 (not by the non-silence check alone) — replaced
    # with versions verified clean via spectral check, see SOURCES.md.
    "city": "city_ambient_clean.mp3",
    # Procedural rain was rejected on listen ("not gonna work... very
    # generic") 2026-08-05 — overrides the procedural generator below via
    # generate_audio_track's dispatch order (sourced checked first).
    "rain": "rain_heavy_deep.mp3",
    "rain_hawaii": "rain_jungle.mp3",  # tropical/rainforest variant, for the Hawaii theme specifically
}

METADATA_TEMPLATES = {
    "white": {"title": "White Noise for Sleep, Focus & Tinnitus Relief", "desc_focus": "study, work, or block out distractions"},
    "pink": {"title": "Pink Noise for Deep Sleep & Relaxation", "desc_focus": "fall asleep faster and stay asleep"},
    "brown": {"title": "Brown Noise for Deep Sleep & Focus", "desc_focus": "a deep, rumbling background for sleep or focus"},
    "red": {"title": "Red Noise for Deep Sleep", "desc_focus": "a deep, echoing background for sleep"},
    "green": {"title": "Green Noise for Focus & Relaxation", "desc_focus": "a balanced, natural-feeling background sound"},
    "rain": {"title": "Rain Sounds for Sleep & Relaxation", "desc_focus": "gentle rainfall to help you relax or sleep"},
    "wind": {"title": "Wind Sounds for Sleep & Relaxation", "desc_focus": "gentle wind ambience for relaxation"},
    "thunder": {"title": "Thunderstorm Sounds for Sleep", "desc_focus": "distant thunder and rain for deep sleep"},
    "birds": {"title": "Morning Birdsong for Relaxation & Focus", "desc_focus": "peaceful birdsong to start your day or help you focus"},
    "cafe": {"title": "Cafe Ambience for Studying & Focus", "desc_focus": "coffee shop chatter for studying or working"},
    "city": {"title": "City Ambience for Sleep & Focus", "desc_focus": "distant urban sounds for sleep or focus"},
}

PROCEDURAL_NOISE = set(generators.NOISE_TYPES)
PROCEDURAL_AMBIENCE = set(ambience.AMBIENCE_TYPES)


def _decode_audio(path, sample_rate=SAMPLE_RATE):
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), tmp_path],
            check=True,
        )
        sr, samples = wavfile.read(tmp_path)
        return sr, samples.astype(np.float64) / 32768.0
    finally:
        os.unlink(tmp_path)


def make_loopable(samples, sample_rate, crossfade_sec=3.0):
    """Crossfades the tail into the head so the clip can repeat back-to-back
    without a click at the seam. See video_pipeline module docstring."""
    n = len(samples)
    cf = int(crossfade_sec * sample_rate)
    if cf * 2 >= n:
        cf = max(1, n // 4)
    head, tail, body = samples[:cf], samples[n - cf:], samples[cf:n - cf]
    fade_out = np.linspace(1, 0, cf)
    fade_in = np.linspace(0, 1, cf)
    blended = tail * fade_out + head * fade_in
    return np.concatenate([blended, body])


def render_sourced_long_form(source_path, total_duration_sec, out_path, crossfade_sec=3.0):
    sr, samples = _decode_audio(source_path)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.9
    loop_unit = make_loopable(samples, sr, crossfade_sec)
    target_n = int(total_duration_sec * sr)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        written = 0
        while written < target_n:
            remaining = target_n - written
            chunk = loop_unit[:remaining] if remaining < len(loop_unit) else loop_unit.copy()
            if written == 0:
                fade_n = min(int(0.05 * sr), len(chunk))
                chunk[:fade_n] *= np.linspace(0, 1, fade_n)
            if written + len(chunk) >= target_n:
                fade_n = min(int(0.05 * sr), len(chunk))
                chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
            wf.writeframes(np.int16(np.clip(chunk, -1, 1) * 32767).tobytes())
            written += len(chunk)
    return out_path


def generate_audio_track(sound_type, duration_sec, out_path, seed=None):
    # Sourced checked first so an explicit override (like rain, see
    # SOURCED_FILES) wins over the procedural generator for the same name.
    if sound_type in SOURCED_FILES:
        source_path = os.path.join(SOURCED_DIR, SOURCED_FILES[sound_type])
        return render_sourced_long_form(source_path, duration_sec, out_path)
    if sound_type in PROCEDURAL_NOISE:
        return generators.render_long_form(sound_type, duration_sec, out_path, seed=seed)
    if sound_type in PROCEDURAL_AMBIENCE:
        return ambience.render_long_form(sound_type, duration_sec, out_path, seed=seed)
    raise ValueError(f"Unknown sound type: {sound_type}")


def get_visual_loop(preset_name, loop_duration_sec=60, width=1920, height=1080, fps=15):
    # fps=15 not 30: motion is slow ambient drift, the difference isn't
    # perceptible, and it roughly halves render time on top of the
    # bounding-box particle optimization (~30min vs ~4.7hrs for all 11
    # presets at 1080p, benchmarked 2026-07-31).
    from visuals.generator import render_loop
    cache_path = os.path.join(VISUALS_DIR, "cache", f"{preset_name}_{width}x{height}.mp4")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        render_loop(preset_name, loop_duration_sec, fps, width, height, cache_path)
    return cache_path


def assemble_video(audio_path, visual_loop_path, out_path, duration_sec):
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-stream_loop", "-1", "-i", str(visual_loop_path),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration_sec), "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def generate_metadata(sound_type, duration_sec, channel_name="Drift"):
    hours = duration_sec / 3600
    if hours >= 1:
        length_label = f"{hours:.0f} Hour" + ("s" if round(hours) != 1 else "")
    else:
        minutes = max(1, round(duration_sec / 60))
        length_label = f"{minutes} Minute" + ("s" if minutes != 1 else "")

    tpl = METADATA_TEMPLATES.get(sound_type, {"title": f"{sound_type.title()} Ambience", "desc_focus": "background ambience"})
    sound_label = f"{sound_type} noise" if sound_type in PROCEDURAL_NOISE else f"{sound_type} sounds"
    title = f"{length_label} of {tpl['title']}"
    description = (
        f"{length_label} of {tpl['title'].lower()} — {tpl['desc_focus']}.\n\n"
        f"No mid-video ads, just uninterrupted {sound_label}.\n\n"
        f"#{sound_type}noise #ambience #sleep #focus"
    )
    tags = [sound_type, "ambience", "sleep", "focus", "relaxation", "study", f"{sound_type} noise", channel_name.lower()]
    return {"title": title, "description": description, "tags": tags}


def produce_video(sound_type, duration_sec, out_dir=None, visual_preset=None, video_resolution=(1920, 1080)):
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "renders")
    os.makedirs(out_dir, exist_ok=True)
    visual_preset = visual_preset or sound_type

    audio_path = os.path.join(out_dir, f"{sound_type}_{int(duration_sec)}s_audio.wav")
    generate_audio_track(sound_type, duration_sec, audio_path)

    width, height = video_resolution
    visual_loop_path = get_visual_loop(visual_preset, width=width, height=height)

    video_path = os.path.join(out_dir, f"{sound_type}_{int(duration_sec)}s.mp4")
    assemble_video(audio_path, visual_loop_path, video_path, duration_sec)

    metadata = generate_metadata(sound_type, duration_sec)
    return {"video_path": video_path, "audio_path": audio_path, "metadata": metadata}
