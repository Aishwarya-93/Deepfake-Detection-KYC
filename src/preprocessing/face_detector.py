"""
Implementation of face detection using OpenCV Haar Cascade classifier.

Author: Tanvi
"""

import cv2
from pathlib import Path

def load_face_detector():
    """
    Load OpenCV Haar Cascade face detector.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")

    return face_cascade

def detect_faces(image, face_cascade):
    """
    Detect faces in an image.

    Returns:
        faces: list of bounding boxes
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    return faces

def draw_faces(image, faces):
    """
    Draw rectangles around detected faces.
    """

    output = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    return output

def crop_faces(image, faces):
    """
    Crop detected faces from the image.
    """

    cropped = []

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cropped.append(face)

    return cropped

from pathlib import Path

def save_faces(faces, output_dir):
    """
    Save cropped faces.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, face in enumerate(faces):
        filename = output_dir / f"face_{i}.jpg"

        cv2.imwrite(
            str(filename),
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        )

    print(f"Saved {len(faces)} face(s).")