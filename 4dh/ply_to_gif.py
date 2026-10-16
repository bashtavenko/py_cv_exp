"""
Renders a 360-degree turntable animation of a 3D mesh from a PLY file and saves it as an animated GIF.
"""
import os

import imageio
import numpy as np
import open3d as o3d
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("input_ply", "output/output_4d_smpl.ply", "Path to input PLY mesh file.")
flags.DEFINE_string("output_gif", "output/turntable.gif", "Path to save output animated GIF.")
flags.DEFINE_integer("num_frames", 60, "Number of frames for a full 360-degree rotation.")


def main(argv):
    del argv

    if not os.path.exists(FLAGS.input_ply):
        raise FileNotFoundError(f"Could not find PLY file at {FLAGS.input_ply}")

    logging.info("Loading mesh from %s...", FLAGS.input_ply)
    mesh = o3d.io.read_triangle_mesh(FLAGS.input_ply)
    if mesh.is_empty():
        raise ValueError("The loaded mesh file is empty.")

    mesh.compute_vertex_normals()

    width, height = 150, 150
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    vis.reset_view_point(True)

    frames = []
    logging.info("Rendering %d turntable frames...", FLAGS.num_frames)

    angle_step = (2.0 * np.pi) / FLAGS.num_frames

    for i in range(FLAGS.num_frames):
        R = mesh.get_rotation_matrix_from_xyz((0.0, angle_step, 0.0))
        mesh.rotate(R, center=mesh.get_center())

        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        img_np = (255.0 * np.asarray(image)).astype("uint8")
        frames.append(img_np)

    vis.destroy_window()

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output_gif)), exist_ok=True)
    logging.info("Saving animated GIF to %s...", FLAGS.output_gif)
    imageio.mimsave(FLAGS.output_gif, frames, fps=20, loop=0)
    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
