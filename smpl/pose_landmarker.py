import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from absl import app, flags, logging
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from smpl_lift import fit_smpl_from_keypoints

FLAGS = flags.FLAGS
flags.DEFINE_string("input_image", "testdata/person.jpeg", "Human image.")
flags.DEFINE_string("output_image", "output/person_keypoints.jpg", "Annotated output image.")


def resource_path(relative_path: str) -> str:
    """
    Provides an embedded resource path per
    https://developers.google.com/edge/mediapipe/solutions/setup_python#packaging_python_tasks_apps_with_pyinstaller
    Need pyinstaller for the deployment per this doc.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# BlazePose 33-landmark skeleton (pairs of keypoint indices)
POSE_CONNECTIONS = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm / right arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg / right leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def draw_and_save_keypoints(
        image_path: str,
        keypoints_2d: np.ndarray,
        output_path: str,
        draw_skeleton: bool = True,
) -> None:
    """Draws 2D keypoints (pixel coordinates) on the image and saves it."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    pts = np.round(keypoints_2d).astype(int)

    if draw_skeleton:
        for i, j in POSE_CONNECTIONS:
            cv2.line(image, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise IOError(f"Failed to write image to {output_path}")
    logging.info("Saved annotated image to %s", output_path)


def extract_2d_keypoints(image_path: str):
    model_path = resource_path("pose_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    logging.info("PoseLandmarker created")

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    height, width, _ = cv_image.shape
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Recreate detector on every call
    with vision.PoseLandmarker.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in image.")

    keypoints_2d = np.array([
        [lm.x * width, lm.y * height]
        for lm in detection_result.pose_landmarks[0]
    ])
    return keypoints_2d


def main(argv):
    del argv
    kps = extract_2d_keypoints(FLAGS.input_image)
    logging.info("Extracted %d 2D keypoints.", len(kps))
    draw_and_save_keypoints(FLAGS.input_image, kps, FLAGS.output_image)
    fit_smpl_from_keypoints(kps)


if __name__ == "__main__":
    app.run(main)
