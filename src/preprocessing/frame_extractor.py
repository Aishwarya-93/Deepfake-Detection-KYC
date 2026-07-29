"""
frame_extractor.py

Reusable functions for extracting frames from videos.

Author: Shreya
"""

from pathlib import Path
import cv2


def get_video_info(video_path):
    """
    Get basic information about a video.

    Parameters
    ----------
    video_path : str or Path

    Returns
    -------
    dict
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video:\n{video_path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    cap.release()

    return info


def extract_frames(video_path, output_dir, interval=10):
    """
    Extract every nth frame from a video.

    Parameters
    ----------
    video_path : str or Path
    output_dir : str or Path
    interval : int

    Returns
    -------
    dict
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video:\n{video_path}")

    frame_number = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:

            filename = output_dir / f"frame_{saved_count:04d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    return {
        "frames_read": frame_number,
        "frames_saved": saved_count,
        "interval": interval,
    }