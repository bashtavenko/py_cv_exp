"""
2D Kalman filter with alternating measurements based on textbook Figure 19.5.

The true state evolves in an approximate circle. We assume a random walk
temporal model. The measurement model is non-stationary:
- At even time steps (k=0, 2, 4...), we measure ONLY the x-coordinate (w_1).
- At odd time steps (k=1, 3, 5...), we measure ONLY the y-coordinate (w_2).

Usage:
    python 2d_example_2.py
    python 2d_example_2.py --show
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

matplotlib.use("qtagg")

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_samples", 15, "Number of position samples.")
flags.DEFINE_float("dt", 1.0, "Sample interval (s).")
flags.DEFINE_float("radius", 10.0, "Radius of the true circular trajectory.")
flags.DEFINE_float("sigma_model", 1.5,
                   "Process noise std (random walk perturbation).")
flags.DEFINE_float("sigma_meas", 1.0, "Measurement noise std.")
flags.DEFINE_string("output_dir", "output", "Directory for saved plots.")
flags.DEFINE_boolean("show", False, "Show plots interactively.")
flags.DEFINE_integer("seed", 42,
                     "Random seed for measurements (-1 = unseeded).")


def generate_true_trajectory(n_samples, radius):
    """Generates a true state that evolves in a 2D circle."""
    # Match the approximate 3/4 circle from the textbook figure
    angles = np.linspace(0.2, 1.8 * np.pi, n_samples)
    x_true = radius * np.cos(angles)
    y_true = radius * np.sin(angles)
    return np.vstack((x_true, y_true))


def run_kalman_alternating(true_states, sigma_model, sigma_meas, rng):
    """Run the Kalman filter with alternating 1D measurements."""
    n_samples = true_states.shape[1]

    # Process Model Matrices (Random Walk)
    phi = np.eye(2)  # State transition
    q = np.eye(2) * (sigma_model ** 2)  # Process noise covariance
    r = np.array([[sigma_meas ** 2]])  # Measurement noise is now 1D scalar

    # Initial Guesses
    x_k_prev = true_states[:, 0:1]  # Start near the truth
    p = np.eye(2) * (sigma_meas ** 2)

    x_k_buffer = np.zeros((2, n_samples))
    p_buffer = np.zeros((2, 2, n_samples))
    m_history = []  # Keep track of which axis was measured for plotting colors

    for k in range(n_samples):
        # 1. Prediction Step
        x_pred = phi @ x_k_prev
        p_pred = phi @ p @ phi.T + q

        # 2. Alternating Measurement Model
        if k % 2 == 0:
            # Measure X only (w_1)
            m = np.array([[1.0, 0.0]])
            z = true_states[0, k] + sigma_meas * rng.standard_normal()
            m_history.append("x")
        else:
            # Measure Y only (w_2)
            m = np.array([[0.0, 1.0]])
            z = true_states[1, k] + sigma_meas * rng.standard_normal()
            m_history.append("y")

        # 3. Update Step
        s = m @ p_pred @ m.T + r  # S is a 1x1 scalar here
        k_gain = p_pred @ m.T @ np.linalg.inv(s)

        x_k = x_pred + k_gain @ (np.array([[z]]) - m @ x_pred)
        p = p_pred - k_gain @ m @ p_pred

        # 4. Save state
        x_k_buffer[:, k] = x_k.ravel()
        p_buffer[:, :, k] = p
        x_k_prev = x_k

    return x_k_buffer, p_buffer, m_history


def get_cov_ellipse(cov, pos, nstd=1, **kwargs):
    """Returns a matplotlib Ellipse patch for the given 2D covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues/vectors to get the primary axis
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Calculate angle and dimensions
    theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(eigvals)

    return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)


def plot_alternating_measurements(true_states, x_k_buffer, p_buffer, m_history,
                                  output_path, show):
    """Plots the true state and Kalman estimates with error ellipses."""
    n_samples = true_states.shape[1]
    fig, ax = plt.subplots(figsize=(8, 8))

    # True state (red dots/lines)
    ax.plot(true_states[0, :], true_states[1, :], "r.-", markersize=8, zorder=3,
            label="True state")

    # Connecting black lines for the estimates
    ax.plot(x_k_buffer[0, :], x_k_buffer[1, :], "k-", zorder=1, linewidth=1.5)

    # Plot Estimates and Covariance Ellipses
    for k in range(n_samples):
        pos = (x_k_buffer[0, k], x_k_buffer[1, k])
        cov = p_buffer[:, :, k]

        # Color coding:
        # Blue when X was measured (ellipse stretches vertically along Y)
        # Cyan when Y was measured (ellipse stretches horizontally along X)
        color = "blue" if m_history[k] == "x" else "teal"

        # Plot estimate dot
        ax.scatter(pos[0], pos[1], color=color, zorder=4, s=30)

        # Plot covariance ellipse (using 1.5 standard deviations for visibility)
        ellip = get_cov_ellipse(cov, pos, nstd=1.5, edgecolor=color,
                                facecolor="none",
                                linestyle="--", linewidth=1.2, zorder=2)
        ax.add_patch(ellip)

    ax.set_title("Kalman Filter with Alternating 1D Measurements")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$", rotation=0, labelpad=15)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color="red", marker=".", label="True State"),
        Line2D([0], [0], color="blue", marker="o", linestyle="--",
               label="Posterior (measured $w_1$)"),
        Line2D([0], [0], color="teal", marker="o", linestyle="--",
               label="Posterior (measured $w_2$)")
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Pad limits so ellipses aren't cut off
    ax.set_xlim(np.min(true_states[0, :]) - 5, np.max(true_states[0, :]) + 5)
    ax.set_ylim(np.min(true_states[1, :]) - 5, np.max(true_states[1, :]) + 5)

    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info("Saved to %s", path)
    if show:
        plt.show()
    plt.close(fig)


def main(argv):
    del argv
    rng = np.random.default_rng(None if FLAGS.seed < 0 else FLAGS.seed)
    logging.info("Running alternating 2D Kalman filter: n_samples=%d",
                 FLAGS.n_samples)

    true_states = generate_true_trajectory(FLAGS.n_samples, FLAGS.radius)

    x_k_buffer, p_buffer, m_history = run_kalman_alternating(
        true_states, FLAGS.sigma_model, FLAGS.sigma_meas, rng)

    output_dir = Path(FLAGS.output_dir)
    plot_alternating_measurements(
        true_states, x_k_buffer, p_buffer, m_history,
        output_dir / "kalman_2d_example_2.png", FLAGS.show)


if __name__ == "__main__":
    app.run(main)
