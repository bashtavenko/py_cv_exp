"""
Kalman filter intuitive tutorial based on
https://www.mathworks.com/matlabcentral/fileexchange/13479-an-intuitive
-introduction-to-kalman-filter/files/Tutorial.m

Predict the position and velocity of a moving train 2 seconds ahead, having
noisy measurements of its positions along the previous 10 seconds.

Ground truth: train starts at x = 0 and moves with constant velocity V = 10 m/s,
so after 12 seconds x = 120 m.

Usage:
    python kalman/kalman.py
    python kalman/kalman.py --show
    python kalman/kalman.py --output_dir=output --sigma_meas=1.0
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

matplotlib.use("qtagg")

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_samples", 100, "Number of position samples.")
flags.DEFINE_float("dt", 0.1, "Sample interval (s).")
flags.DEFINE_float("v_true", 10.0, "True constant velocity (m/s).")
flags.DEFINE_float("sigma_model", 1.0, "Initial model uncertainty std.")
flags.DEFINE_float("sigma_meas", 1.0, "Measurement noise std (m).")
flags.DEFINE_integer("window_size", 5, "Running-average window for velocity.")
flags.DEFINE_integer("samples_into_future", 20,
                     "Extrapolation horizon (samples).")
flags.DEFINE_integer("n_last", 10, "Samples used for extrapolation plot.")
flags.DEFINE_string("output_dir", "output", "Directory for saved plots.")
flags.DEFINE_boolean("show", False, "Show plots interactively.")
flags.DEFINE_integer("seed", -1,
                     "Random seed for measurements (-1 = unseeded).")


def moving_average(x, window):
    """MATLAB filter(ones(1, window) / window, 1, x)."""
    b = np.ones(window) / window
    y = np.zeros(len(x), dtype=float)
    for n in range(len(x)):
        seg = x[max(0, n - len(b) + 1): n + 1]
        y[n] = np.dot(b[-len(seg):], seg)
    return y


def run_kalman(t, x_true, dt, v_true, sigma_model, sigma_meas, rng):
    """Run the Kalman filter and return state/measurement buffers."""
    n_samples = len(t) - 1

    x_k_prev = np.array([[0.0], [0.5 * v_true]])
    phi = np.array([[1.0, dt], [0.0, 1.0]])
    p = np.array([[sigma_model ** 2, 0.0], [0.0, sigma_model ** 2]])
    q = np.zeros((2, 2))
    m = np.array([[1.0, 0.0]])
    r = sigma_meas ** 2

    x_k_buffer = np.zeros((2, n_samples + 1))
    x_k_buffer[:, 0] = x_k_prev.ravel()
    z_buffer = np.zeros(n_samples + 1)

    for k in range(n_samples):
        z = x_true[k + 1] + sigma_meas * rng.standard_normal()
        z_buffer[k + 1] = z

        p1 = phi @ p @ phi.T + q
        s = m @ p1 @ m.T + r
        # np.linalg.solve(s, ...) is numerically preferable for larger S;
        # inv() matches
        k_gain = p1 @ m.T @ np.linalg.inv(s)
        p = p1 - k_gain @ m @ p1

        x_k = phi @ x_k_prev + k_gain * (z - m @ phi @ x_k_prev)
        x_k_buffer[:, k + 1] = x_k.ravel()
        x_k_prev = x_k

    return x_k_buffer, z_buffer


def save_figure(fig, path, show):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info("Saved to %s", path)
    if show:
        plt.show()
    plt.close(fig)


def plot_position(t, x_true, z_buffer, x_k_buffer, output_path, show):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, x_true, "g", label="True position")
    ax.plot(t, z_buffer, "c", label="Measurements")
    ax.plot(t, x_k_buffer[0, :], "m", label="Kalman estimated displacement")
    ax.set_title("Position estimation results")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_path, show)


def plot_velocity(t, v_true, z_buffer, x_k_buffer, dt, window_size, output_path,
                  show):
    n_samples = len(t) - 1
    instantaneous_velocity = np.zeros(n_samples + 1)
    instantaneous_velocity[1:] = (z_buffer[1:] - z_buffer[:-1]) / dt
    instantaneous_velocity_avg = moving_average(instantaneous_velocity,
                                                window_size)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, np.full_like(t, v_true), "m", label="True velocity")
    ax.plot(t, instantaneous_velocity, "g",
            label="Estimated velocity by raw consecutive samples")
    ax.plot(t, instantaneous_velocity_avg, "c",
            label="Estimated velocity by running average")
    ax.plot(t, x_k_buffer[1, :], "k",
            label="Estimated velocity by Kalman filter")
    ax.set_title("Velocity estimation results")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_path, show)
    return instantaneous_velocity_avg


def plot_extrapolation(n_samples, dt, v_true, x_k_buffer, velocity_avg,
                       samples_into_future, n_last, output_path, show):
    true_position = (n_samples + samples_into_future) * v_true * dt
    # MATLAB (Nsamples+1-Nlast):(Nsamples+1) is inclusive on both ends.
    start = n_samples + 1 - n_last
    stop = n_samples + 2
    idx = slice(start, stop)
    horizons = np.arange(samples_into_future + n_last, samples_into_future - 1,
                         -1)

    projected_by_avg = (
            x_k_buffer[0, idx]
            + horizons * dt * velocity_avg[idx]
    )
    projected_by_kalman = (
            x_k_buffer[0, idx]
            + horizons * dt * x_k_buffer[1, idx]
    )
    sample_times = np.arange(start, stop) * dt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_times, np.full(len(sample_times), true_position), "m",
            label="True position")
    ax.plot(sample_times, projected_by_avg, "c",
            label="Estimated position by running average")
    ax.plot(sample_times, projected_by_kalman, "k",
            label="Estimated position by Kalman filter")
    target_t = (n_samples + samples_into_future) * dt
    ax.set_title(f"Extrapolation 20 samples ahead (at t = {target_t:g})")
    ax.set_xlabel("Time of sample used for extrapolation (s)")
    ax.set_ylabel("Expected position (m)")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_path, show)


def main(argv):
    del argv
    n_samples = FLAGS.n_samples
    dt = FLAGS.dt
    v_true = FLAGS.v_true

    t = np.arange(0.0, dt * (n_samples + 1), dt)
    x_true = v_true * t

    rng = np.random.default_rng(None if FLAGS.seed < 0 else FLAGS.seed)
    logging.info("Running Kalman filter: n_samples=%d, dt=%.1f, v_true=%.1f",
                 n_samples, dt, v_true)
    x_k_buffer, z_buffer = run_kalman(
        t, x_true, dt, v_true, FLAGS.sigma_model, FLAGS.sigma_meas, rng)

    output_dir = Path(FLAGS.output_dir)
    plot_position(t, x_true, z_buffer, x_k_buffer,
                  output_dir / "kalman_position.png", FLAGS.show)
    velocity_avg = plot_velocity(
        t, v_true, z_buffer, x_k_buffer, dt, FLAGS.window_size,
        output_dir / "kalman_velocity.png", FLAGS.show)
    plot_extrapolation(
        n_samples, dt, v_true, x_k_buffer, velocity_avg,
        FLAGS.samples_into_future, FLAGS.n_last,
        output_dir / "kalman_extrapolation.png", FLAGS.show)


if __name__ == "__main__":
    app.run(main)
