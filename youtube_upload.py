"""YouTube upload automation via the Data API v3.

Auth uses the OAuth Desktop-app credentials from account setup
(credentials/youtube_client_secret.json). The Cloud Console app is still
in "Testing" publish status, so the refresh token expires every 7 days —
get_authenticated_service() re-triggers the browser consent flow whenever
the saved token is missing/expired rather than failing silently. Moving
to production (removing the 7-day expiry) requires Google's app
verification process — not done yet, revisit if the weekly re-auth
becomes annoying.
"""

import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
BASE_DIR = os.path.dirname(__file__)
CLIENT_SECRET_PATH = os.path.join(BASE_DIR, "credentials", "youtube_client_secret.json")

# The Google account has two Brand Account channels (Drift Noise, Drift
# Sounds), and a single OAuth grant is tied to whichever channel was
# active in the browser at consent time (confirmed 2026-07-31 — the
# first token turned out to authorize Sounds, not Noise as assumed).
# So each channel needs its own token file, and each needs its own
# consent flow run while that specific channel is active in the browser.
TOKEN_PATHS = {
    "noise": os.path.join(BASE_DIR, "credentials", "token_noise.pickle"),
    "sounds": os.path.join(BASE_DIR, "credentials", "token_sounds.pickle"),
}

CHANNEL_MAP = {
    "white": "noise", "pink": "noise", "brown": "noise", "red": "noise", "green": "noise",
    "rain": "sounds", "wind": "sounds", "thunder": "sounds", "birds": "sounds", "cafe": "sounds", "city": "sounds",
}

# Mid-roll ad placement isn't exposed on videos.insert/update for regular
# uploads (confirmed 2026-07-07) — it's a channel-level Studio toggle
# ("automatic ad breaks"), and moot until the channel clears YPP anyway.
# Nothing to set here for that.
DEFAULT_CATEGORY_ID = "22"  # People & Blogs


def get_authenticated_service(channel):
    if channel not in TOKEN_PATHS:
        raise ValueError(f"Unknown channel: {channel}. Choose from {list(TOKEN_PATHS)}")
    token_path = TOKEN_PATHS[channel]

    creds = None
    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None  # refresh token itself expired (7-day Testing-mode limit) -> re-auth below

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
            print(f"\nMake sure Drift {channel.title()} is the ACTIVE channel in your browser, then open this URL:\n")
            creds = flow.run_local_server(port=0, open_browser=False)

        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


def upload_video(youtube, video_path, metadata, privacy_status="private",
                  thumbnail_path=None, category_id=DEFAULT_CATEGORY_ID):
    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "tags": metadata.get("tags", []),
            "categoryId": category_id,
        },
        "status": {"privacyStatus": privacy_status},
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response["id"]

    thumbnail_set = False
    if thumbnail_path:
        try:
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
            thumbnail_set = True
        except Exception as e:
            # Custom thumbnails need the channel to be phone-verified — don't
            # let that (or any other thumbnail hiccup) take down an otherwise
            # successful video upload. Caller can check thumbnail_set and
            # retry youtube.thumbnails().set(...) later once verified.
            print(f"  (thumbnail not set, continuing without it: {e})")

    return video_id, thumbnail_set


def generate_thumbnail(preset_name, out_path, title_text=None, width=1280, height=720):
    from visuals.generator import render_frame, PRESETS
    cfg = PRESETS[preset_name]
    frame = render_frame(
        0.3, width, height,
        palette=cfg["palette"],
        particle_color=cfg.get("particle_color"),
        particle_style=cfg.get("particle_style", "drift"),
        n_particles=cfg.get("n_particles", 40),
        particle_speed=cfg.get("particle_speed", 1.0),
    )
    img = Image.fromarray(frame)

    if title_text:
        draw = ImageDraw.Draw(img)
        font = None
        for candidate in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                           "/System/Library/Fonts/Helvetica.ttc"]:
            if os.path.exists(candidate):
                font = ImageFont.truetype(candidate, size=int(height * 0.09))
                break
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (width - tw) / 2, height * 0.75 - th / 2
        # simple outline for readability over a variable-brightness background
        for ox, oy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
            draw.text((x + ox, y + oy), title_text, font=font, fill=(0, 0, 0))
        draw.text((x, y), title_text, font=font, fill=(255, 255, 255))

    img.save(out_path)
    return out_path
