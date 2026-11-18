"""
Boilerplate of Kalman filtering
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # ============================================================
    # 1. System Parameters (Constant Velocity Object)
    # ============================================================
    dt = 0.1

    a = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])

    b = np.array([
        [0.5 * dt ** 2],
        [dt]
    ])

    h = np.array([[1.0, 0.0]])  # Measure position only

    # ============================================================
    # 2. Noise Covariances
    # ============================================================
    q = np.array([
        [0.01, 0.0],
        [0.0, 0.01]
    ])

    r = np.array([[0.5]])

    # ============================================================
    # 3. Generate Simulated Noisy Truth Data
    # ============================================================
    np.random.seed(42)

    t = np.arange(0.0, 10.0 + dt, dt)
    n = len(t)

    true_pos = 0.5 * t ** 2  # accelerating trajectory

    noise = np.sqrt(r[0, 0]) * np.random.randn(n)
    measurements = true_pos + noise

    # Optional dataframe for inspection
    df = pd.DataFrame({
        "time_s": t,
        "true_position": true_pos,
        "measurement": measurements
    })
    print(df)

    # ============================================================
    # 4. Initialize Kalman Filter Variables
    # ============================================================
    x_est = np.array([
        [0.0],  # position
        [0.0]  # velocity
    ])

    p = np.eye(2)

    saved_states = np.zeros((2, n))

    # ============================================================
    # 5. Recursive Kalman Filter Loop
    # ============================================================
    u = 1.0  # constant acceleration input

    for k in range(n):
        # -------------------------
        # Prediction Step
        # -------------------------
        x_pred = a @ x_est + b * u
        p_pred = a @ p @ a.T + q

        # -------------------------
        # Correction Step
        # -------------------------
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)

        y = np.array([[measurements[k]]])

        innovation = y - h @ x_pred

        x_est = x_pred + k @ innovation
        p = (np.eye(2) - k @ h) @ p_pred

        saved_states[:, k] = x_est.flatten()

    # ============================================================
    # 6. Plot Results
    # ============================================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        t,
        measurements,
        "r.",
        label="Noisy Measurements"
    )

    plt.plot(
        t,
        true_pos,
        "k-",
        linewidth=1.5,
        label="True Trajectory"
    )

    plt.plot(
        t,
        saved_states[0, :],
        "b-",
        linewidth=1.5,
        label="Kalman Estimate"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Linear Kalman Filter Tracking")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
