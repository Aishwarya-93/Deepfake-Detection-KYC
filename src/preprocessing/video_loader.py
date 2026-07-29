"""
video_loader.py

Author: Aishwarya Lakshmi
Project: Deepfake Detection for Video KYC

Utility functions for loading and inspecting videos
from the FaceForensics++ dataset.
"""

import cv2
import matplotlib.pyplot as plt


def load_video(video_path: str):
    """
    Opens a video file using OpenCV.

    Parameters:
        video_path (str or Path): Path to the video.

    Returns:
        cv2.VideoCapture: VideoCapture object.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    return cap


def is_video_valid(video_path: str) -> bool:
    """
    Checks whether a video exists and can be opened.
    """

    cap = cv2.VideoCapture(str(video_path))
    valid = cap.isOpened()
    cap.release()

    return valid


def get_video_info(cap) -> dict:
    """
    Returns basic information about the video.
    """

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = round(frame_count / fps, 2) if fps > 0 else 0

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height
    }


def read_first_frame(cap):
    """
    Reads the first frame of a video.

    Returns:
        tuple: (success, frame)
    """

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, frame = cap.read()

    return success, frame


def display_frame(frame, title: str = "Frame"):
    """
    Displays a frame using matplotlib.
    """

    if frame is None:
        print("No frame to display.")
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def close_video(cap) -> None:
    """
    Releases the video object.
    """

    if cap is not None:
        cap.release()