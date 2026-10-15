"""
4D human mesh inference using a pre-trained HMR2.0 model.
"""
from hmr2.datasets.vitdet_dataset import ViTDetDataset
# Assume that Conda environment is set up and activated
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from absl import app, flags, logging
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS
flags.DEFINE_string("input_image", "testdata/person3.jpeg", "Human image.")
flags.DEFINE_string("output_mesh", "output/output_4d_smpl.ply", "Output mesh.")
flags.DEFINE_boolean("show_mesh", True, "Visualize the generated 3D mesh using Open3D.")


def run_4d_inference(image_path: str, output_mesh_path: str):
    logging.info("Loading pre-trained HMR2.0 model from 4D-Humans...")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model.eval().cuda()

    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    height, width, _ = cv_img.shape

    boxes = np.array([[0, 0, width, height]], dtype=np.float32)
    dataset = ViTDetDataset(model_cfg, cv_img, boxes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    logging.info("Running forward pass...")
    for batch in dataloader:
        batch = recursive_to(batch, 'cuda')
        with torch.no_grad():
            output = model(batch)

        # Extract predicted vertices and SMPL faces
        pred_vertices = output['pred_vertices'].detach().cpu().numpy()[0]  # (6890, 3)
        faces = model.smpl.faces  # (13776, 3)

        # Export mesh to PLY
        mesh = trimesh.Trimesh(
            vertices=pred_vertices,
            faces=faces,
            process=False
        )
        mesh.export(output_mesh_path)
        break

    logging.info("Mesh successfully saved to %s", output_mesh_path)


def main(argv):
    del argv

    run_4d_inference(FLAGS.input_image, FLAGS.output_mesh)
    if FLAGS.show_mesh:
        logging.info("Opening Open3D viewer...")
        mesh = o3d.io.read_triangle_mesh(FLAGS.output_mesh)

        if mesh.is_empty():
            logging.error("Could not load mesh file for visualization.")
        else:
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="4D-Humans SMPL 3D Mesh Viewer",
                width=1024,
                height=768
            )


if __name__ == "__main__":
    app.run(main)
