"""
`2D Kalman filter tutorial based on Simon J.D. Prince Computer Vision textbook
Figure 19.4.

Object approximately circling around a central point in two dimensions
(x_t = x_{t-1} + noise) with the Brownian motion model (randowm walk).
The true state evolves in a 2D circle.
the filter still successfully reduces posterior covariance compared to raw
measurements.

Usage:
    python kalman_2d.py
    python kalman_2d.py --show
    python kalman_2d.py --output_dir=output --sigma_meas=2.0
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

matplotlib.use("qtagg")

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_samples", 20, "Number of position samples.")
flags.DEFINE_float("dt", 1.0, "Sample interval (s).")
flags.DEFINE_float("radius", 10.0, "Radius of the true circular trajectory.")
flags.DEFINE_float("sigma_model", 1.5,
                   "Process noise std (random walk perturbation).")
flags.DEFINE_float("sigma_meas", 2.5, "Measurement noise std.")
flags.DEFINE_string("output_dir", "output", "Directory for saved plots.")
flags.DEFINE_boolean("show", False, "Show plots interactively.")
flags.DEFINE_integer("seed", 42,
                     "Random seed for measurements (-1 = unseeded).")


def generate_true_trajectory(n_samples, radius):
    """Generates a true state that evolves in a 2D circle."""
    angles = np.linspace(0, 1.5 * np.pi, n_samples)
    x_true = radius * np.cos(angles)
    y_true = radius * np.sin(angles)
    return np.vstack((x_true, y_true))


def run_kalman_2d(true_states, sigma_model, sigma_meas, rng):
    """Run the 2D Kalman filter assuming a random walk model."""
    n_samples = true_states.shape[1]

    # Model Matrices (Random Walk: Next state is previous state + noise)
    phi = np.eye(2)  # State transition matrix
    q = np.eye(2) * (sigma_model ** 2)  # Process noise covariance
    m = np.eye(2)  # Measurement matrix
    r = np.eye(2) * (sigma_meas ** 2)  # Measurement noise covariance

    # Initial Guesses
    # Start the estimate exactly at the first measurement to match standard
    # priors
    z_initial = true_states[:, 0:1] + sigma_meas * rng.standard_normal((2, 1))
    x_k_prev = z_initial
    p = np.eye(2) * (
            sigma_meas ** 2)  # Initial covariance matches measurement noise

    x_k_buffer = np.zeros((2, n_samples))
    z_buffer = np.zeros((2, n_samples))
    p_buffer = np.zeros((2, 2, n_samples))

    for k in range(n_samples):
        # 1. Generate Noisy Measurement (if k=0, use our initial z)
        if k == 0:
            z = z_initial
        else:
            z = true_states[:, k:k + 1] + sigma_meas * rng.standard_normal(
                (2, 1))

        z_buffer[:, k] = z.ravel()

        # 2. Predict (Time Update)
        x_pred = phi @ x_k_prev
        p_pred = phi @ p @ phi.T + q

        # 3. Update (Measurement Update)
        s = m @ p_pred @ m.T + r
        k_gain = p_pred @ m.T @ np.linalg.inv(s)

        x_k = x_pred + k_gain @ (z - m @ x_pred)
        p = p_pred - k_gain @ m @ p_pred

        # 4. Save state for plotting
        x_k_buffer[:, k] = x_k.ravel()
        p_buffer[:, :, k] = p
        x_k_prev = x_k

    return x_k_buffer, z_buffer, p_buffer


def save_figure(fig, path, show):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info("Saved to %s", path)
    if show:
        plt.show()
    plt.close(fig)


def plot_textbook_comparison(true_states, z_buffer, x_k_buffer, p_buffer,
                             sigma_meas, output_path, show):
    """Plots a side-by-side comparison matching textbook Figure 19.4."""
    n_samples = true_states.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True,
                                   sharey=True)

    # --- Figure A: Measurements Alone ---
    ax1.set_title("a) Measurements Only")
    ax1.set_xlabel("$w_1$")
    ax1.set_ylabel("$w_2$", rotation=0, labelpad=15)

    # True state (red dots/lines)
    ax1.plot(true_states[0, :], true_states[1, :], "r.-", markersize=8,
             label="True state")

    # Measurements (magenta dots, black connecting lines)
    ax1.plot(z_buffer[0, :], z_buffer[1, :], "k-", zorder=1, linewidth=1)
    ax1.scatter(z_buffer[0, :], z_buffer[1, :], color="magenta", zorder=2, s=20,
                label="Measurements")

    # Measurement covariance circles
    for k in range(n_samples):
        circle = plt.Circle((z_buffer[0, k], z_buffer[1, k]), radius=sigma_meas,
                            color="magenta", fill=False, linestyle=":",
                            linewidth=1)
        ax1.add_patch(circle)

    # --- Figure B: Kalman Filter Estimates ---
    ax2.set_title("b) Kalman Filter Posterior")
    ax2.set_xlabel("$w_1$")
    ax2.set_ylabel("$w_2$", rotation=0, labelpad=15)

    # True state (red dots/lines)
    ax2.plot(true_states[0, :], true_states[1, :], "r.-", markersize=8,
             label="True state")

    # Kalman Estimates (teal dots, black connecting lines)
    ax2.plot(x_k_buffer[0, :], x_k_buffer[1, :], "k-", zorder=1, linewidth=1)
    ax2.scatter(x_k_buffer[0, :], x_k_buffer[1, :], color="teal", zorder=2,
                s=20, label="Posterior Mean")

    # Posterior covariance circles
    for k in range(n_samples):
        # Since p_buffer is a diagonal matrix (isotropic), std dev is sqrt of
        # variance
        radius = np.sqrt(p_buffer[0, 0, k])
        circle = plt.Circle((x_k_buffer[0, k], x_k_buffer[1, k]), radius=radius,
                            color="teal", fill=False, linestyle=":",
                            linewidth=1)
        ax2.add_patch(circle)

    # Format axes to look like textbook
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    plt.tight_layout()
    save_figure(fig, output_path, show)


def main(argv):
    del argv

    rng = np.random.default_rng(None if FLAGS.seed < 0 else FLAGS.seed)

    logging.info("Running 2D Kalman filter: n_samples=%d", FLAGS.n_samples)

    true_states = generate_true_trajectory(FLAGS.n_samples, FLAGS.radius)

    x_k_buffer, z_buffer, p_buffer = run_kalman_2d(
        true_states, FLAGS.sigma_model, FLAGS.sigma_meas, rng)

    output_dir = Path(FLAGS.output_dir)
    plot_textbook_comparison(
        true_states, z_buffer, x_k_buffer, p_buffer, FLAGS.sigma_meas,
        output_dir / "kalman_2d_textbook.png", FLAGS.show)


if __name__ == "__main__":
    app.run(main)
