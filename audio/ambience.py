"""Procedural ambient sound layers for Drift Sounds.

These are layers, not necessarily standalone videos — thunder in
particular is meant to be mixed with rain for a "thunderstorm" scene,
same way wind might get mixed with rain for "windy rain," etc. That
mixing happens in the video pipeline stage, not here.

rain    - continuous filtered hiss + sparse droplet transients
wind    - slow crossfade between a calm and a brighter filtered noise
          band, so it gusts without clicking at block boundaries
thunder - sparse, randomly-timed bass-heavy rumbles with a sharp
          attack and long decay
"""

import wave

import numpy as np
from scipy import signal

SAMPLE_RATE = 44100


def _scale_peak(x, peak=0.9):
    x = x - np.mean(x)
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val * peak
    return x


def _apply_fade(x, sample_rate=SAMPLE_RATE, fade_sec=0.05, fade_in=True, fade_out=True):
    fade_n = int(fade_sec * sample_rate)
    if fade_n > 0 and len(x) > 2 * fade_n:
        if fade_in:
            x[:fade_n] *= np.linspace(0, 1, fade_n)
        if fade_out:
            x[-fade_n:] *= np.linspace(1, 0, fade_n)
    return x


def rain(duration_sec, sample_rate=SAMPLE_RATE, seed=None, intensity=0.5, time_offset=0.0):
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)

    white = rng.standard_normal(n)
    sos = signal.butter(4, [1000, 9000], btype="band", fs=sample_rate, output="sos")
    hiss = signal.sosfilt(sos, white)
    hiss = hiss / np.max(np.abs(hiss)) * 0.5

    droplets = np.zeros(n)
    n_drops = int(duration_sec * 40 * intensity)
    drop_len = int(0.008 * sample_rate)
    for pos in rng.integers(0, n, n_drops):
        if pos + drop_len >= n:
            continue
        env = np.exp(-np.linspace(0, 8, drop_len))
        droplets[pos:pos + drop_len] += rng.standard_normal(drop_len) * env * rng.uniform(0.3, 1.0)

    sos2 = signal.butter(2, 4000, btype="low", fs=sample_rate, output="sos")
    droplets = signal.sosfilt(sos2, droplets)

    return _scale_peak(hiss + droplets)


def wind(duration_sec, sample_rate=SAMPLE_RATE, seed=None, gust_freq=0.03, time_offset=0.0):
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    white = rng.standard_normal(n)

    sos_calm = signal.butter(2, 250, btype="low", fs=sample_rate, output="sos")
    sos_gust = signal.butter(2, 900, btype="low", fs=sample_rate, output="sos")
    calm = signal.sosfilt(sos_calm, white)
    gust = signal.sosfilt(sos_gust, white)

    # time_offset keeps the gust cycle continuous across chunk boundaries
    # on a long render — same reasoning as brown noise's wave swell.
    t = time_offset + np.arange(n) / sample_rate
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * gust_freq * t)
    out = calm * (1 - lfo) + gust * lfo
    out = out * (0.6 + 0.4 * lfo)
    return _scale_peak(out)


def thunder(duration_sec, sample_rate=SAMPLE_RATE, seed=None, rumbles_per_min=3, time_offset=0.0):
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    out = np.zeros(n)

    n_rumbles = max(1, int(duration_sec / 60 * rumbles_per_min))
    for pos in rng.integers(0, n, n_rumbles):
        rumble_len = int(rng.uniform(2.0, 5.0) * sample_rate)
        rumble_len = min(rumble_len, n - pos)
        if rumble_len <= 0:
            continue
        raw = rng.standard_normal(rumble_len)
        sos = signal.butter(4, 120, btype="low", fs=sample_rate, output="sos")
        rumble = signal.sosfilt(sos, raw)

        attack_len = min(int(0.05 * sample_rate), rumble_len)
        env = np.ones(rumble_len)
        env[:attack_len] = np.linspace(0, 1, attack_len)
        env[attack_len:] = np.exp(-np.linspace(0, 4, rumble_len - attack_len))
        out[pos:pos + rumble_len] += rumble * env * rng.uniform(0.6, 1.0)

    return _scale_peak(out)


AMBIENCE_TYPES = {
    "rain": rain,
    "wind": wind,
    "thunder": thunder,
}


def render_long_form(ambience_type, total_duration_sec, out_path, sample_rate=SAMPLE_RATE,
                      seed=None, chunk_sec=600):
    """Same chunked-generation approach as generators.render_long_form —
    see that docstring for why chunk_sec needs to stay well above any
    texture period (wind's gust is the longest here, ~33s)."""
    if ambience_type not in AMBIENCE_TYPES:
        raise ValueError(f"Unknown ambience type: {ambience_type}. Choose from {list(AMBIENCE_TYPES)}")
    fn = AMBIENCE_TYPES[ambience_type]
    base_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        elapsed = 0.0
        chunk_idx = 0
        while elapsed < total_duration_sec - 1e-9:
            this_len = min(chunk_sec, total_duration_sec - elapsed)
            samples = fn(this_len, sample_rate, seed=base_seed + chunk_idx, time_offset=elapsed)
            is_first = chunk_idx == 0
            is_last = elapsed + this_len >= total_duration_sec - 1e-9
            samples = _apply_fade(samples, sample_rate, fade_in=is_first, fade_out=is_last)
            wf.writeframes(np.int16(samples * 32767).tobytes())
            elapsed += this_len
            chunk_idx += 1
    return out_path
