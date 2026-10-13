"""
Extracts keypoints and saves them to a file.
"""
import cv2
from absl import app, flags, logging

from smplify.body_keypoints import extract_2d_keypoints, draw_and_save_keypoints

FLAGS = flags.FLAGS
flags.DEFINE_string("input_image", "testdata/person3.jpeg", "Human image.")
flags.DEFINE_string("output_image", "output/person_keypoints.jpg", "Annotated output image.")


def main(argv):
    cv_image = cv2.imread(FLAGS.input_image)
    if cv_image is None:
        raise FileNotFoundError(f"Could not load image at {FLAGS.input_image}")
    height, width, _ = cv_image.shape
    kps = extract_2d_keypoints(cv_image, height, width)
    logging.info("Extracted %d 2D keypoints.", len(kps))
    draw_and_save_keypoints(FLAGS.input_image, kps, FLAGS.output_image)


if __name__ == "__main__":
    app.run(main)
