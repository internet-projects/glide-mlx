# Copied unchanged from fastino/GLiDE @ d199b6d7 (glide/images.py), Fastino Community License (see LICENSE).
"""Image decoding and preprocessing for the vision encoder.

Images are resized so both sides are multiples of 32 pixels with the pixel count kept inside [GLIDE_MIN_PIXELS, GLIDE_MAX_PIXELS]
(aspect ratio preserved), normalised with mean = std = 0.5 and cut into 16 x 16 patches (2 x 2 patches merge into one visual token).
With the default GLIDE_MAX_PIXELS = 1024 * 1024 an image uses at most 1,024 visual tokens.
"""
from __future__ import annotations

import base64
import io
import math
import os

MIN_PIXELS = int(os.environ.get("GLIDE_MIN_PIXELS", str(256 * 256)))
MAX_PIXELS = int(os.environ.get("GLIDE_MAX_PIXELS", str(1024 * 1024)))


def load_image(x):
    """PIL image, raw bytes, base64 string or data URL -> RGB PIL image."""
    from PIL import Image
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    if not isinstance(x, str):
        raise ValueError("image must be a data URL or a base64 string")
    if x.startswith("data:"):
        x = x.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")


def smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Both sides multiples of factor, pixel count inside [min_pixels, max_pixels], aspect ratio kept."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def image_inputs(img, patch=16, temporal=2, merge=2, min_pixels=None, max_pixels=None):
    """RGB PIL image -> (flattened patches [N, 3 * temporal * patch * patch] float16, (1, grid_h, grid_w), visual token count)."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    hb, wb = smart_resize(h, w, patch * merge, min_pixels or MIN_PIXELS, max_pixels or MAX_PIXELS)
    x = np.asarray(img.resize((wb, hb), Image.BICUBIC), dtype=np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]                      # (1, C, H, W)
    x = np.repeat(x, temporal, axis=0)                                  # (temporal, C, H, W)
    gh, gw = hb // patch, wb // patch
    x = x.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * temporal * patch * patch)
    return x.astype(np.float16), (1, gh, gw), gh * gw // (merge * merge)
