"""Makes a non-looping video clip (e.g. raw Sora output, which tends to
have a narrative arc rather than a steady state) into a seamless loop,
using the same idea as audio/video_pipeline.py's make_loopable: blend the
tail into the head instead of hard-cutting, so frame(end) flows into
frame(start) instead of jumping.

ffmpeg's xfade filter does the actual blend; this just orchestrates
extracting the pieces and reassembling them.
"""

import json
import os
import subprocess
import tempfile


def _probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def make_video_loopable(source_path, out_path, crossfade_sec=1.5, fps=24):
    duration = _probe_duration(source_path)
    cf = crossfade_sec
    if cf * 2 >= duration:
        raise ValueError(f"crossfade_sec ({cf}) too long for a {duration:.1f}s clip")

    with tempfile.TemporaryDirectory() as tmp:
        tail_path = os.path.join(tmp, "tail.mp4")
        head_path = os.path.join(tmp, "head.mp4")
        middle_path = os.path.join(tmp, "middle.mp4")
        blended_path = os.path.join(tmp, "blended.mp4")
        concat_list = os.path.join(tmp, "concat.txt")

        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), "-an"]

        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source_path,
                         "-ss", str(duration - cf), "-t", str(cf), *enc, tail_path], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source_path,
                         "-ss", "0", "-t", str(cf), *enc, head_path], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source_path,
                         "-ss", str(cf), "-t", str(duration - 2 * cf), *enc, middle_path], check=True)

        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tail_path, "-i", head_path,
                         "-filter_complex", f"[0][1]xfade=transition=fade:duration={cf}:offset=0",
                         *enc, blended_path], check=True)

        with open(concat_list, "w") as f:
            f.write(f"file '{blended_path}'\nfile '{middle_path}'\n")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                         "-i", concat_list, "-c", "copy", out_path], check=True)

    return out_path
