"""Measure interior EM orientation and fine-to-coarse texture energy."""
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

METRICS = ("em_texture_directionality", "em_texture_fine_fraction")


def texture(image, foreground):
    # The largest Gaussian kernel extends four voxels per axis. Keep its
    # entire support inside the object and the downloaded crop.
    valid = minimum_filter(foreground, size=9, mode="constant", cval=0)
    if np.count_nonzero(valid) < 64:
        return np.nan, np.nan
    smooth = [gaussian_filter(image, sigma, truncate=2)[valid]
              for sigma in (0.5, 1, 2)]
    grad = np.stack([
        gaussian_filter(image, 1, order=order, truncate=2)[valid]
        for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(grad @ grad.T)
    total = eigenvalues.sum()
    direction = (1.5 * eigenvalues[-1] / total - 0.5) if total > 0 else np.nan
    fine, coarse = [np.mean(np.square(a - b)) for a, b in zip(smooth, smooth[1:])]
    return direction, fine / (fine + coarse) if fine + coarse > 0 else np.nan
