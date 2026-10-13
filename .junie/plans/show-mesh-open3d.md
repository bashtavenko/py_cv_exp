---
sessionId: session-261013-110842-1rip
---

# Requirements

### Overview & Goals
Provide an interactive 3D mesh visualization capability in `smplify/two_stage_main.py` using Open3D. This maximizes developer utility and inspection ergonomics by enabling immediate visual feedback on SMPL fitting results directly after optimization without requiring external 3D viewer software.

### Scope
- **In Scope**:
  - Adding `open3d` to `smplify/requirements.txt`.
  - Adding a CLI boolean flag (`--show_mesh`, default: `False`) to `smplify/two_stage_main.py`.
  - Adding mesh loading and interactive rendering using Open3D (`open3d.visualization.draw_geometries`) after fitting is complete.
  - Ensuring vertex normals are computed so the mesh renders with proper shading.
- **Out of Scope**:
  - Modifying the underlying SMPL loss formulation or optimization loop in `smplify/smpl_lift.py`.
  - Headless off-screen rendering or GUI framework replacements.

### User Stories
- As a computer vision engineer running `two_stage_main.py`, I want to pass `--show_mesh` so that I can inspect the resulting 3D SMPL human mesh in an interactive Open3D window.
- As an automated pipeline user running `two_stage_main.py` in batch/CI mode without a display, I want `--show_mesh` to default to `False` so that headless execution proceeds without blocking or GUI errors.

### Functional Requirements
- **CLI Flag Definition**: `smplify/two_stage_main.py` defines `flags.DEFINE_bool("show_mesh", False, "Whether to display the 3D mesh interactively using Open3D.")` (or equivalent `DEFINE_boolean`).
- **Conditional Visualization**: When `FLAGS.show_mesh` is `True`, the script opens an interactive Open3D window displaying the generated `.ply` mesh after `fit_smpl_from_keypoints` completes.
- **Mesh Shading**: Ensure vertex normals are computed on the loaded `TriangleMesh` before rendering to provide clear 3D surface depth and geometry contours.

### Non-Functional Requirements
- **Efficiency & Utility**: Minimal performance overhead when visualization is disabled; instant launch when enabled.
- **Error Resilience**: Provide clear error messaging if Open3D fails to initialize or display on headless environments.


# Technical Design

### Current Implementation
`smplify/two_stage_main.py` currently defines flags for `--input_image` and `--output_mesh`, extracts 2D keypoints using MediaPipe, and optimizes SMPL parameters via `fit_smpl_from_keypoints(kps, height, width, FLAGS.output_mesh)`. The resulting mesh is exported to a `.ply` file using `trimesh`, but there is no built-in viewer to inspect the 3D output immediately.

### Key Decisions
- **Flag Definition via `absl.flags`**: Use `flags.DEFINE_bool("show_mesh", False, ...)` to remain consistent with existing absl flag conventions in `two_stage_main.py` and `body_keypoints_main.py`.
- **Decoupled Visualization Function**: Implement `visualize_mesh_open3d(mesh_path: str)` in `smplify/smpl_lift.py` (or directly within `smplify/two_stage_main.py`) using `open3d.io.read_triangle_mesh` and `mesh.compute_vertex_normals()` before `open3d.visualization.draw_geometries([mesh])`.
- **Default Value**: Set default to `False` to prevent blocking batch jobs, scripts, or headless environments.

### Proposed Changes
1. **`smplify/requirements.txt`**:
   - Append `open3d` to the project dependencies list.
2. **`smplify/smpl_lift.py`**:
   - Add a helper function `visualize_mesh_open3d(mesh_path: str)`:
     ```python
     def visualize_mesh_open3d(mesh_path: str) -> None:
         """Loads a 3D mesh from disk and displays it using Open3D."""
         import open3d as o3d
         mesh = o3d.io.read_triangle_mesh(mesh_path)
         if mesh.is_empty():
             logging.error("Failed to load mesh from %s for visualization.", mesh_path)
             return
         mesh.compute_vertex_normals()
         logging.info("Opening Open3D visualization window...")
         o3d.visualization.draw_geometries([mesh], window_name="SMPL 3D Mesh")
     ```
3. **`smplify/two_stage_main.py`**:
   - Add flag: `flags.DEFINE_bool("show_mesh", False, "Whether to visualize the resulting mesh with Open3D.")`.
   - In `main()`:
     ```python
     fit_smpl_from_keypoints(kps, height, width, FLAGS.output_mesh)
     if FLAGS.show_mesh:
         visualize_mesh_open3d(FLAGS.output_mesh)
     ```

### Data Models / Contracts
- `flags.DEFINE_bool("show_mesh", False, "Whether to visualize the resulting mesh with Open3D.")`
- `visualize_mesh_open3d(mesh_path: str) -> None`

### File Structure
- `smplify/requirements.txt` (modified: add `open3d`)
- `smplify/smpl_lift.py` (modified: add `visualize_mesh_open3d` helper)
- `smplify/two_stage_main.py` (modified: add `--show_mesh` flag and conditional viewer invocation)

### Risks & Mitigations
- **Headless Display Environments**: Running with `--show_mesh` on a server without X11/Wayland display will raise an Open3D GLFW error.
  *Mitigation*: Default `--show_mesh` to `False` and catch display initialization errors gracefully with a helpful log message.


# Testing

### Validation Approach
Verify the CLI argument handling and visualization trigger under multiple conditions.

### Key Scenarios
1. **Default Execution (`--show_mesh=False` / omitted)**:
   - Run `python smplify/two_stage_main.py`.
   - Verify that SMPL fitting executes, exports `output/output_smpl2.ply`, and finishes without opening a GUI window.
2. **Interactive Visualization (`--show_mesh=True` / `--show_mesh`)**:
   - Run `python smplify/two_stage_main.py --show_mesh`.
   - Verify that the resulting mesh is read by Open3D, vertex normals are computed, and the Open3D viewer window opens displaying the human mesh.
3. **Custom Output Path with Visualization**:
   - Run with custom `--output_mesh /tmp/test_mesh.ply --show_mesh`.
   - Verify that Open3D loads and visualizes the custom output mesh file.

### Edge Cases
- **Missing or Corrupted Mesh File**: Verify that an informative error message is logged if the output mesh file does not exist or fails to load in Open3D.
- **Headless Execution Guard**: Verify that failures in opening GUI display do not crash unhandled exceptions during batch runs.


# Delivery Steps

###   Step 1: Add Open3D dependency and mesh visualization helper
The project declares `open3d` in requirements and provides a robust Open3D mesh rendering helper.

- Add `open3d` to `smplify/requirements.txt`.
- Implement a helper function `visualize_mesh(mesh_path: str)` (or `visualize_mesh(mesh)`) in `smplify/smpl_lift.py` (or a dedicated visualization module) that loads the triangle mesh, computes vertex normals for smooth shading, and opens an interactive Open3D viewer window using `open3d.visualization.draw_geometries`.

###   Step 2: Integrate show_mesh CLI flag in two_stage_main.py
Users can pass a command-line flag to view the fitted 3D SMPL mesh interactively upon completion.

- Define an absl boolean flag `--show_mesh` (defaulting to `False`) in `smplify/two_stage_main.py`.
- Update `main()` in `smplify/two_stage_main.py` to invoke the Open3D visualizer with `FLAGS.output_mesh` when `--show_mesh` is set to `True`.
- Ensure informative logging is displayed when launching the interactive viewer.