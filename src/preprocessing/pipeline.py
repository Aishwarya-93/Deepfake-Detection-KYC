"""
pipeline.py

Integrates:
1. Video Loading
2. Frame Extraction
3. Face Detection
4. Face Cropping
5. Image Preprocessing

Author: Aishwarya
"""

from pathlib import Path
import cv2

from src.preprocessing import frame_extractor
from src.preprocessing import face_detector
from src.preprocessing import image_preprocessing


def process_video(
    video_path,
    frame_output_dir,
    processed_output_dir,
    interval=10
):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    video_path : str or Path
    frame_output_dir : str or Path
    processed_output_dir : str or Path
    interval : int

    Returns
    -------
    dict
    """

    video_path = Path(video_path)
    frame_output_dir = Path(frame_output_dir)
    processed_output_dir = Path(processed_output_dir)

    frame_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading face detector...")

    detector = face_detector.load_face_detector()

    print("Extracting frames...")

    extraction_info = frame_extractor.extract_frames(
        video_path=video_path,
        output_dir=frame_output_dir,
        interval=interval
    )

    frame_files = sorted(
        frame_output_dir.glob("*.jpg")
    )

    print(
        f"Frames found: {len(frame_files)}"
    )

    total_faces = 0

    for frame_file in frame_files:

        image = cv2.imread(str(frame_file))

        if image is None:
            continue

        faces = face_detector.detect_faces(
            image,
            detector
        )

        cropped_faces = face_detector.crop_faces(
            image,
            faces
        )

        for face in cropped_faces:

            try:

                processed_face = (
                    image_preprocessing.resize_image(
                        face,
                        (224, 224)
                    )
                )

                processed_face = (
                    image_preprocessing.normalize_image(
                        processed_face
                    )
                )

                save_path = (
                    processed_output_dir
                    / f"face_{total_faces:05d}.jpg"
                )

                cv2.imwrite(
                    str(save_path),
                    (processed_face * 255).astype("uint8")
                )

                total_faces += 1

            except Exception as e:

                print(
                    f"Error processing face: {e}"
                )

    print(
        f"Pipeline complete. Faces processed: {total_faces}"
    )

    return {
        "frames_extracted": extraction_info[
            "frames_saved"
        ],
        "faces_processed": total_faces
    }


if __name__ == "__main__":

    DATASET = Path(
        "data/raw/FaceForensics++_C23/original"
    )

    video_files = sorted(
        DATASET.glob("*.mp4")
    )

    if len(video_files) == 0:
        raise FileNotFoundError(
            "No videos found."
        )

    test_video = video_files[0]

    results = process_video(
        video_path=test_video,
        frame_output_dir="outputs/frames",
        processed_output_dir="outputs/processed_faces",
        interval=10
    )

    print(results)