"""
build_preprocessed_dataset.py

Builds a processed face-image dataset from FaceForensics++ videos.

Pipeline:
Video
    ↓
Frame extraction
    ↓
Face detection
    ↓
Face cropping
    ↓
Resize to 224x224
    ↓
Save face image

Author: Aishwarya
"""

from pathlib import Path
import cv2

from src.preprocessing.face_detector import (
    load_face_detector,
    detect_faces,
    crop_faces,
)

from src.preprocessing.image_preprocessing import resize_image


def build_processed_dataset(
    dataframe,
    dataset_root,
    output_root,
    split_name,
    frame_interval=10,
    max_frames_per_video=5,
):
    """
    Generate processed face images from videos.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing File Path and Label columns.

    dataset_root : str or Path
        Root directory of FaceForensics++ dataset.

    output_root : str or Path
        Root directory where processed images are saved.

    split_name : str
        Dataset split: train, val, or test.

    frame_interval : int
        Extract one frame every nth frame.

    max_frames_per_video : int
        Maximum number of frames processed from each video.

    Returns
    -------
    dict
        Processing statistics.
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    detector = load_face_detector()

    total_videos = 0
    total_frames = 0
    total_faces = 0
    failed_videos = 0

    for _, row in dataframe.iterrows():

        relative_path = Path(row["File Path"])
        label = str(row["Label"]).strip().lower()

        video_path = dataset_root / relative_path

        if label not in ["real", "fake"]:
            print(f"Skipping unknown label: {label}")
            continue

        output_dir = output_root / split_name / label
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if not video_path.exists():
            print(f"Video not found: {video_path}")
            failed_videos += 1
            continue

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Unable to open: {video_path}")
            failed_videos += 1
            continue

        video_name = video_path.stem

        frame_number = 0
        frames_used = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_number % frame_interval == 0:

                total_frames += 1

                faces = detect_faces(
                    frame,
                    detector
                )

                cropped_faces = crop_faces(
                    frame,
                    faces
                )

                for face_index, face in enumerate(
                    cropped_faces
                ):

                    resized_face = resize_image(
                        face,
                        size=(224, 224)
                    )

                    output_file = (
                        output_dir
                        / f"{video_name}_"
                          f"frame_{frame_number:06d}_"
                          f"face_{face_index}.jpg"
                    )

                    cv2.imwrite(
                        str(output_file),
                        resized_face
                    )

                    total_faces += 1

                frames_used += 1

                if frames_used >= max_frames_per_video:
                    break

            frame_number += 1

        cap.release()

        total_videos += 1

        if total_videos % 10 == 0:
            print(
                f"{split_name}: "
                f"{total_videos}/{len(dataframe)} videos processed"
            )

    return {
        "split": split_name,
        "videos_processed": total_videos,
        "frames_processed": total_frames,
        "faces_saved": total_faces,
        "failed_videos": failed_videos,
    }