from __future__ import annotations

import numpy as np


TARGET_RATE_HZ = 25.0
FRAME_COUNT_CLOSE_TOLERANCE = 25

# Engagement annotations and most cached features are intended to be consumed
# on a 25 Hz grid. The tolerance is deliberately small: it handles tiny tail
# differences between streams and targets without treating real duration
# mismatches as aligned.

def align_to_target_grid(
    mat: np.ndarray,
    source_rate_hz: float,
    target_len: int,
    target_rate_hz: float = TARGET_RATE_HZ,
    frame_count_tolerance: int = FRAME_COUNT_CLOSE_TOLERANCE,
) -> tuple[np.ndarray, str]:
    """Align a stream matrix to the 25 Hz engagement target grid.

    The preferred path is to trust near-identical frame counts, because several
    streams have unreliable sample-rate metadata while their frame counts match
    the target. If counts do not match, the function falls back to linear
    interpolation on the declared source/target rates.
    """

    # Empty streams cannot contribute to a model input. Return an explicit
    # method label so the status/manifests can explain skipped examples.
    if mat.shape[0] == 0 or target_len <= 0:
        return np.empty((0, mat.shape[1]), dtype=np.float32), "empty"

    # Preferred path: if frame counts already match the target closely, keep the
    # existing grid. This protects against known cases where sr metadata is less
    # reliable than the frame count.
    if abs(len(mat) - target_len) <= frame_count_tolerance:
        return mat[: min(len(mat), target_len)].astype(np.float32, copy=False), "truncate_frame_count_close"

    # If the declared rate is already the target rate, only trim to the shared
    # duration. Padding is avoided here because extra invented frames can affect
    # sequence models and validation reconstruction.
    if abs(source_rate_hz - target_rate_hz) < 1e-6:
        return mat[: min(len(mat), target_len)].astype(np.float32, copy=False), "truncate_rate_match"

    # Fallback path: construct source and target time axes and interpolate every
    # feature dimension independently. NaNs are repaired by interpolation within
    # the source stream before resampling to the target grid.
    source_t = np.arange(mat.shape[0], dtype=np.float64) / source_rate_hz
    target_t = np.arange(target_len, dtype=np.float64) / target_rate_hz
    keep = target_t <= source_t[-1]
    if not keep.any():
        return np.empty((0, mat.shape[1]), dtype=np.float32), "empty_after_resample_grid"

    target_t = target_t[keep]
    out = np.empty((len(target_t), mat.shape[1]), dtype=np.float32)
    for col_idx in range(mat.shape[1]):
        col = mat[:, col_idx].astype(np.float64, copy=False)
        finite = np.isfinite(col)
        if not finite.any():
            out[:, col_idx] = np.nan
        elif finite.all():
            out[:, col_idx] = np.interp(target_t, source_t, col).astype(np.float32)
        else:
            repaired = np.interp(source_t, source_t[finite], col[finite])
            out[:, col_idx] = np.interp(target_t, source_t, repaired).astype(np.float32)
    return out, "linear_interpolation"
