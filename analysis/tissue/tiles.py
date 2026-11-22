"""Tile helpers: centre crops, Otsu tissue masks, labelled tiles from region masks (numpy only)."""
import numpy as np


def center_crop(img, size):
    """Central size x size crop of an (H, W, ...) array."""
    h, w = img.shape[:2]
    y, x = h // 2 - size // 2, w // 2 - size // 2
    return img[y:y + size, x:x + size]


def otsu_threshold(gray):
    """Otsu threshold of grey values 0-255 (256-bin histogram)."""
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    between[~np.isfinite(between)] = -1
    return float(np.argmax(between))


def tissue_fraction(rgb, thr):
    """Share of pixels darker than thr in grey (mean of the RGB channels)."""
    gray = np.asarray(rgb, dtype=float)[..., :3].mean(-1)
    return float((gray < thr).mean())


def labelled_tiles(mask, tile, min_cover=0.8):
    """(x, y, code) for non-overlapping full tiles whose most common code covers >= min_cover.
    Code 0 (outside the annotated region) is never returned."""
    out = []
    h, w = mask.shape[:2]
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            counts = np.bincount(mask[y:y + tile, x:x + tile].ravel())
            code = int(np.argmax(counts))
            if code != 0 and counts[code] >= min_cover * tile * tile:
                out.append((x, y, code))
    return out
