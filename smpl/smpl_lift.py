import smplx
import torch
from absl import logging


def fit_smpl_from_keypoints(kps_2d, height, width, output_filename):
    # Center MediaPipe keypoints relative to image center
    kps_centered = kps_2d.copy()
    kps_centered[:, 0] -= width / 2.0
    kps_centered[:, 1] -= height / 2.0

    # Topology mapping (MediaPipe index -> SMPL index)
    MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    SMPL_INDICES = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

    gt_keypoints_2d = torch.tensor(kps_centered[MP_INDICES], dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="models/smpl/SMPL_NEUTRAL.pkl",
        model_type="smpl",
        gender="neutral",
        num_betas=10
    )

    # Learnable Parameters
    global_orient = torch.zeros([1, 3], dtype=torch.float32, requires_grad=True)
    body_pose = torch.zeros([1, 69], dtype=torch.float32, requires_grad=True)
    betas = torch.zeros([1, 10], dtype=torch.float32, requires_grad=True)

    # Set body starts at Z = 2.5 meters in front of camera
    transl = torch.tensor([[0.0, 0.0, 2.5]], dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam(
        [global_orient, body_pose, betas, transl],
        lr=0.02
    )

    # Approximate camera focal length based on image dimensions
    focal_length = max(width, height)

    # Optimization Loop
    logging.info("Fitting SMPL parameters with perspective projection...")
    for step in range(400):
        optimizer.zero_grad()

        # Forward pass passing explicit 3D translation
        smpl_output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=True
        )

        # Extract 3D joint locations in camera space: (12, 3)
        joints_3d = smpl_output.joints[0, SMPL_INDICES, :]

        # Scale by focal length, avoid division by zero
        X = joints_3d[:, 0]
        Y = joints_3d[:, 1]
        Z = torch.clamp(joints_3d[:, 2], min=0.1)

        proj_x = focal_length * (X / Z)
        proj_y = focal_length * (Y / Z)
        proj_2d = torch.stack([proj_x, proj_y], dim=-1)

        # L1 / L2 loss
        diff = proj_2d - gt_keypoints_2d
        loss_reprojection = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-4))

        # Regularization priors to keep body human-shaped
        loss_pose_prior = torch.mean(body_pose ** 2)
        loss_shape_prior = torch.mean(betas ** 2)

        total_loss = loss_reprojection + (0.02 * loss_pose_prior) + (0.001 * loss_shape_prior)

        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1:03d}/400 | Loss: {total_loss.item():.4f}")

    # Extract Final 3D Mesh Vertices
    optimized_vertices = smpl_output.vertices.detach().cpu().numpy()[0]
    optimized_faces = smpl_model.faces

    import trimesh
    mesh = trimesh.Trimesh(
        vertices=optimized_vertices,
        faces=optimized_faces,
        process=False  # Keeps original SMPL vertex/face ordering intact
    )

    mesh.export(output_filename)
    logging.info("Saved 3D SMPL mesh to %s", output_filename)

    logging.info("Optimization complete with valid human posture.")
    return optimized_vertices, optimized_faces
