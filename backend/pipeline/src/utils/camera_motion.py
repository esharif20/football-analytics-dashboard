"""Camera motion estimation via sparse optical flow."""

from typing import List

import cv2
import numpy as np
import numpy.typing as npt


def estimate_camera_motions(
    frames: List[npt.NDArray[np.uint8]],
    downscale: float = 0.5,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 30,
) -> List[npt.NDArray[np.float64]]:
    """Estimate per-frame camera motion via sparse optical flow + homography.

    Returns a list of 3×3 homography matrices, one per consecutive frame pair.
    ``motions[j]`` maps pixel coordinates in frame *j* to frame *j+1*
    (forward warp).

    Args:
        frames: List of BGR video frames (full resolution).
        downscale: Factor to shrink frames before flow estimation (speed).
        max_corners: Max Shi-Tomasi corners to detect per frame.
        quality_level: Shi-Tomasi quality level.
        min_distance: Minimum pixel distance between corners.

    Returns:
        List of 3×3 numpy arrays (length = len(frames) - 1).
        Identity matrix is returned for any pair where estimation fails.
    """
    if len(frames) < 2:
        return []

    motions: List[npt.NDArray[np.float64]] = []
    inv_scale = 1.0 / downscale

    # Pre-convert first frame
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    if downscale != 1.0:
        prev_gray = cv2.resize(
            prev_gray, None, fx=downscale, fy=downscale,
            interpolation=cv2.INTER_AREA,
        )

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for j in range(len(frames) - 1):
        curr_gray = cv2.cvtColor(frames[j + 1], cv2.COLOR_BGR2GRAY)
        if downscale != 1.0:
            curr_gray = cv2.resize(
                curr_gray, None, fx=downscale, fy=downscale,
                interpolation=cv2.INTER_AREA,
            )

        # Detect corners in previous frame
        corners = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=max_corners,
            qualityLevel=quality_level, minDistance=min_distance,
        )

        if corners is None or len(corners) < 4:
            motions.append(np.eye(3, dtype=np.float64))
            prev_gray = curr_gray
            continue

        # Forward optical flow
        pts_next, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None, **lk_params,
        )

        # Backward consistency check
        pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, pts_next, None, **lk_params,
        )

        # Filter: both flows succeeded + forward-backward error < 1 px
        fb_error = np.linalg.norm(
            corners.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1,
        )
        mask = (
            (status_fwd.ravel() == 1)
            & (status_bwd.ravel() == 1)
            & (fb_error < 1.0)
        )

        src = corners.reshape(-1, 2)[mask]
        dst = pts_next.reshape(-1, 2)[mask]

        if len(src) < 4:
            motions.append(np.eye(3, dtype=np.float64))
            prev_gray = curr_gray
            continue

        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

        if H is None:
            motions.append(np.eye(3, dtype=np.float64))
        else:
            # Scale homography back to original resolution
            # S @ H_small @ S_inv  where S = diag(inv_scale, inv_scale, 1)
            if downscale != 1.0:
                S = np.diag([inv_scale, inv_scale, 1.0])
                S_inv = np.diag([downscale, downscale, 1.0])
                H = S @ H @ S_inv
            motions.append(H)

        prev_gray = curr_gray

    return motions


def warp_keypoints(
    keypoints: npt.NDArray[np.float32],
    cumulative_H: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Warp 2D keypoint positions using a homography matrix.

    Args:
        keypoints: (N, 2) array of keypoint positions.
        cumulative_H: 3×3 homography matrix.

    Returns:
        (N, 2) array of warped keypoint positions.
    """
    if keypoints.size == 0:
        return keypoints.copy()
    reshaped = keypoints.reshape(-1, 1, 2).astype(np.float32)
    H_f32 = cumulative_H.astype(np.float64)
    warped = cv2.perspectiveTransform(reshaped, H_f32)
    return warped.reshape(-1, 2).astype(np.float32)
