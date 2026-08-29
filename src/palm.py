import cv2
import numpy as np


def extract_palm(frame, keypoints, padding=0.25):
    """
    Extract the palm region from a detected hand.

    keypoints:
        21 hand landmarks in (x, y) format.

    Returns:
        Cropped palm image, or None if extraction fails.
    """

    if keypoints is None or len(keypoints) != 21:
        return None

    # Palm landmarks:
    # 0  = wrist
    # 5  = index finger MCP
    # 9  = middle finger MCP
    # 13 = ring finger MCP
    # 17 = pinky MCP

    palm_points = keypoints[[0, 5, 9, 13, 17]]

    x_min = int(np.min(palm_points[:, 0]))
    y_min = int(np.min(palm_points[:, 1]))
    x_max = int(np.max(palm_points[:, 0]))
    y_max = int(np.max(palm_points[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    # Add padding around the palm.
    pad_x = int(width * padding)
    pad_y = int(height * padding)

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    h, w = frame.shape[:2]

    # Keep crop inside image boundaries.
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    palm = frame[y_min:y_max, x_min:x_max]

    if palm.size == 0:
        return None

    return palm


def draw_palm_box(frame, keypoints, padding=0.25):
    """
    Draw the palm extraction region for debugging.
    """

    if keypoints is None or len(keypoints) != 21:
        return frame

    palm_points = keypoints[[0, 5, 9, 13, 17]]

    x_min = int(np.min(palm_points[:, 0]))
    y_min = int(np.min(palm_points[:, 1]))
    x_max = int(np.max(palm_points[:, 0]))
    y_max = int(np.max(palm_points[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    pad_x = int(width * padding)
    pad_y = int(height * padding)

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    h, w = frame.shape[:2]

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "PALM REGION",
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    return frame