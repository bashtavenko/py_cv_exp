"""Lift 2D BlazePose keypoints to a 3D SMPL body via SMPLify-style optimization."""

import torch
import smplx


def fit_smpl_from_keypoints(kps_2d):
    # Center around origin and scale to [-1, 1]
    norm_kps_2d = (kps_2d - 0.5) * 2.0

    # 1. Define Topology Mapping (MediaPipe index -> SMPL index)
    # MediaPipe: [L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle]
    MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    SMPL_INDICES = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

    # Filter 2D target keypoints to match mapped topology
    gt_keypoints_2d = torch.tensor(norm_kps_2d[MP_INDICES], dtype=torch.float32)

    smpl_model = smplx.create(
        model_path="models/smpl/SMPL_NEUTRAL.pkl",
        model_type="smpl",
        gender="neutral",
        num_betas=10
    )

    # Initialize Learnable Parameters (Tensors requiring gradients)
    global_orient = torch.zeros([1, 3], dtype=torch.float32, requires_grad=True)  # Root rotation
    body_pose = torch.zeros([1, 69], dtype=torch.float32, requires_grad=True)  # 23 joint axis-angles
    betas = torch.zeros([1, 10], dtype=torch.float32, requires_grad=True)  # Body shape

    # Weak perspective camera parameters: Scale s and Translation (tx, ty)
    cam_scale = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    cam_trans = torch.zeros([1, 2], dtype=torch.float32, requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        [global_orient, body_pose, betas, cam_scale, cam_trans],
        lr=0.02
    )

    # Optimization Loop
    print("Fitting SMPL parameters to 2D keypoints...")
    for step in range(300):
        optimizer.zero_grad()

        # Forward pass: Generate 3D joints from current SMPL parameters
        smpl_output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            return_verts=True
        )

        # Extract 3D joint locations corresponding to mapped indices: (12, 3)
        joints_3d = smpl_output.joints[0, SMPL_INDICES, :]

        # Weak Perspective Projection: P_2d = scale * P_3d[:, :2] + trans
        proj_2d = cam_scale * joints_3d[:, :2] + cam_trans

        # Loss Functions:
        # a. Reprojection Error (MSE between 2D target and projected SMPL 3D joints)
        loss_reprojection = torch.mean((proj_2d - gt_keypoints_2d) ** 2)

        # b. Pose & Shape Regularization Priors (prevents unnatural joint bending)
        loss_pose_prior = torch.mean(body_pose ** 2)
        loss_shape_prior = torch.mean(betas ** 2)

        # Combined Objective Function
        total_loss = loss_reprojection + (0.01 * loss_pose_prior) + (0.001 * loss_shape_prior)

        # Backpropagate and update parameters
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1:03d}/300 | Loss: {total_loss.item():.6f}")

    #  Extract Final 3D Mesh Vertices
    optimized_vertices = smpl_output.vertices.detach().numpy()[0]  # (6890, 3)
    optimized_faces = smpl_model.faces  # (13776, 3)

    print("\nOptimization Complete!")
    print(f"Optimized Mesh Vertices shape: {optimized_vertices.shape}")

