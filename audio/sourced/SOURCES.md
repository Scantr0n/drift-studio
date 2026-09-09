# Sourced Audio — Provenance & License

All files below are from Mixkit (mixkit.co), under the Mixkit Sound Effects Free
License: free for commercial use including YouTube, no attribution required.
License text: https://mixkit.co/license/

| File | Source title | Source ID | Duration |
|---|---|---|---|
| birds_morning.mp3 | Morning birds | 2472 | 3:29 |
| birds_forest.mp3 | Forest birds ambience | 1210 | 2:31 |
| cafe_restaurant_chatter.mp3 | Restaurant crowd talking ambience | 444 | 2:00 |
| city_night.mp3 | Urban city ambience at night | 2678 | 2:06 |
| city_day.mp3 | Urban ambience during the day | 2505 | 0:46 |
| rain_jungle.mp3 | Jungle rain and birds | 2392 | 1:20 |
| city_ambient_clean.mp3 | Urban ambient sound | 2465 | 1:05 |
| rain_heavy_deep.mp3 | Heavy rain ambience | 1262 | 2:09 |
| waterfall.mp3 | Waterfall in the woods | 2517 | 1:22 |
| wind_forest.mp3 | Wind in the forest | 1237 | 0:25 |
| rain_on_glass.mp3 | Heavy rain on car glass interior | 1248 | 0:40 |
| owl_forest.mp3 | Owl in a forest | 2466 | 1:30 |

Downloaded 2026-07-05 (birds/cafe/city), 2026-08-06 (rain — procedural rain was rejected as "not sounding like rain, too generic," switched to real recordings same as birds/cafe/city). rain_jungle.mp3 is specifically for the Hawaii rainforest theme.

**2026-08-16 replacement:** `city_night.mp3` and the original `rain_ambience.mp3` were both dropped — Jack heard birds/crickets in what was supposed to be city/rain audio. Checked with a spectral analysis (2-8kHz tonal "peakiness" — real chirping shows as high-variance pulsing energy in that band, plain broadband hiss doesn't) and confirmed both had it baked into the original field recording, something the original non-silence-only verification missed. `city_ambient_clean.mp3` and `rain_heavy_deep.mp3` replace them — both checked clean with the same method before being selected this time, and `rain_heavy_deep.mp3` specifically has ~500x more low-frequency (20-200Hz) energy than the old rain_ambience.mp3, addressing the separate "not deep enough" complaint. Old `city_night.mp3` and `rain_ambience.mp3` files are still on disk but no longer referenced by the pipeline — fine to delete, kept for now in case of an A/B re-check.

Lesson for future sourcing: always run the spectral contamination check (peakiness in 2-8kHz) before selecting a track, not just a non-silence/duration check — a file can be genuinely non-silent, correct duration, and still not match its label.
