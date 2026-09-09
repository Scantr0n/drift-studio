"""Colored noise generators for Drift Noise.

Each color gets a distinct spectral shape (FFT amplitude shaping) AND a
distinct textural treatment, so they're perceptually different from each
other, not just different on paper:

  white  - pure, unshaped reference static (no texture layer)
  pink   - soft: low-pass smoothed for a gentler, rounder feel
  brown  - deep + wavy: slow amplitude swells, like ocean waves rolling in
  red    - deep + echo: cavernous decaying echo (same base spectrum as
           brown, but a completely different listening experience)
  green  - choppy: faster, irregular organic texture (rustling/insect-like)
"""

import wave

import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 44100


def _fft_noise(duration_sec, sample_rate, shape_fn, seed=None):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    white = rng.standard_normal(n_samples)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC

    shaped = spectrum * shape_fn(freqs)
    return np.fft.irfft(shaped, n=n_samples)


def _scale_peak(x, peak=0.9):
    """Amplitude scaling only — safe to call per-chunk since chunks are far
    longer than any texture period, so peak stays statistically consistent
    chunk to chunk. Fades are handled separately (see _apply_fade) since
    those must only happen at the true start/end of a render, not at every
    internal chunk boundary."""
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


def apply_wave_swell(x, sample_rate=SAMPLE_RATE, swell_freq=0.05, depth=0.35, time_offset=0.0):
    """Slow amplitude swell, like ocean waves rolling in and out (~20s period).

    time_offset lets this be called chunk-by-chunk on a long render without
    the swell resetting to phase 0 at every chunk boundary — pass the
    elapsed seconds so far and the swell stays continuous across the seam.
    """
    t = time_offset + np.arange(len(x)) / sample_rate
    envelope = (1 - depth) + depth * (0.5 + 0.5 * np.sin(2 * np.pi * swell_freq * t))
    return x * envelope


def apply_echo(x, sample_rate=SAMPLE_RATE, delay_sec=0.35, decay=0.55, repeats=5):
    """Cavernous decaying echo — distinct from a plain wave swell."""
    delay_samples = int(delay_sec * sample_rate)
    out = x.copy()
    for i in range(1, repeats + 1):
        shift = delay_samples * i
        if shift >= len(x):
            break
        out[shift:] += x[:-shift] * (decay ** i)
    return out


def apply_soft_smoothing(x, sample_rate=SAMPLE_RATE, cutoff_hz=1800):
    """Low-pass smoothing for a gentler, rounder texture."""
    sos = signal.butter(4, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, x)


def apply_choppy_texture(x, sample_rate=SAMPLE_RATE, rate_hz=2.5, depth=0.5, seed=None):
    """Faster, irregular organic texture — opposite of the slow, smooth wave swell."""
    rng = np.random.default_rng(seed)
    step = max(int(sample_rate / rate_hz), 1)
    n_points = len(x) // step + 2
    raw = rng.random(n_points)
    envelope = np.interp(np.arange(len(x)), np.arange(n_points) * step, raw)
    envelope = (1 - depth) + depth * envelope
    return x * envelope


def white_noise(duration_sec, sample_rate=SAMPLE_RATE, seed=None, time_offset=0.0):
    x = _fft_noise(duration_sec, sample_rate, lambda f: np.ones_like(f), seed)
    return _scale_peak(x)


def pink_noise(duration_sec, sample_rate=SAMPLE_RATE, seed=None, time_offset=0.0):
    x = _fft_noise(duration_sec, sample_rate, lambda f: 1.0 / np.sqrt(f), seed)
    x = apply_soft_smoothing(x, sample_rate)
    return _scale_peak(x)


def brown_noise(duration_sec, sample_rate=SAMPLE_RATE, seed=None, time_offset=0.0):
    x = _fft_noise(duration_sec, sample_rate, lambda f: 1.0 / f, seed)
    x = apply_wave_swell(x, sample_rate, time_offset=time_offset)
    return _scale_peak(x)


def red_noise(duration_sec, sample_rate=SAMPLE_RATE, seed=None, time_offset=0.0):
    # Same base 1/f^2 spectrum as brown noise, but echo instead of a wave
    # swell so it's a genuinely different listening experience, not a
    # relabeled duplicate. Echo is a local operation (no long-period phase),
    # so it doesn't need time_offset continuity across chunks.
    x = _fft_noise(duration_sec, sample_rate, lambda f: 1.0 / f, seed)
    x = apply_echo(x, sample_rate)
    return _scale_peak(x)


def green_noise(duration_sec, sample_rate=SAMPLE_RATE, seed=None,
                center_freq=500.0, bandwidth=400.0, time_offset=0.0):
    def shape(f):
        return np.exp(-0.5 * ((f - center_freq) / bandwidth) ** 2)
    x = _fft_noise(duration_sec, sample_rate, shape, seed)
    x = apply_choppy_texture(x, sample_rate, seed=seed)
    return _scale_peak(x)


NOISE_TYPES = {
    "white": white_noise,
    "pink": pink_noise,
    "brown": brown_noise,
    "red": red_noise,
    "green": green_noise,
}


def render_to_wav(noise_type, duration_sec, out_path, sample_rate=SAMPLE_RATE, seed=None):
    """One-shot short render (samples, previews). For anything long enough
    that holding the whole array in memory is a problem, use render_long_form."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {list(NOISE_TYPES)}")
    samples = NOISE_TYPES[noise_type](duration_sec, sample_rate, seed=seed)
    samples = _apply_fade(samples, sample_rate)
    int_samples = np.int16(samples * 32767)
    wavfile.write(out_path, sample_rate, int_samples)
    return out_path


def render_long_form(noise_type, total_duration_sec, out_path, sample_rate=SAMPLE_RATE,
                      seed=None, chunk_sec=600):
    """Generates arbitrarily long audio (hours) without ever holding more
    than one chunk in memory. Chunk size must stay well above any texture
    period in generators.py (longest is brown's ~20s swell) so per-chunk
    peak scaling doesn't audibly favor a swell peak or trough — 600s gives
    a 30x margin. time_offset keeps swell/gust phase continuous across
    chunk boundaries; fades are only applied to the first/last chunk."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {list(NOISE_TYPES)}")
    fn = NOISE_TYPES[noise_type]
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
