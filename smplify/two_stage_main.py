import cv2
import open3d as o3d
from absl import app, flags, logging

from smplify.body_keypoints import extract_2d_keypoints
from smplify.smpl_lift import fit_smpl_from_keypoints

FLAGS = flags.FLAGS
flags.DEFINE_string("input_image", "testdata/person3.jpeg", "Human image.")
flags.DEFINE_string("output_mesh", "output/output_smpl2.ply", "Output mesh.")
flags.DEFINE_boolean("show_mesh", True, "Visualize the generated 3D mesh using Open3D.")


def main(argv):
    del argv

    cv_image = cv2.imread(FLAGS.input_image)
    if cv_image is None:
        raise FileNotFoundError(f"Could not load image at {FLAGS.input_image}")
    height, width, _ = cv_image.shape
    kps = extract_2d_keypoints(cv_image, height, width)
    logging.info("Extracted %d 2D keypoints.", len(kps))
    fit_smpl_from_keypoints(kps, height, width, FLAGS.output_mesh)

    # See more at https://www.open3d.org/docs/release/tutorial/visualization/visualization.html
    if FLAGS.show_mesh:
        logging.info("Opening Open3D viewer...")
        mesh = o3d.io.read_triangle_mesh(FLAGS.output_mesh)

        if mesh.is_empty():
            logging.error("Could not load mesh file for visualization.")
        else:
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="SMPL 3D Mesh Viewer",
                width=1024,
                height=768
            )


if __name__ == "__main__":
    app.run(main)
