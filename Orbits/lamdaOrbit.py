import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd

# ========= Parameter Definition =========
N = 687
a = 1.524
b = 1.517
e = 0.0934
i = np.radians(1.85)
obliquity = np.radians(25.2)  # Rotational axis tilt
spin_period_step = 1.025957
spin_rate_per_step = 2 * np.pi / spin_period_step

# ========= 1. Orbit Data Generation =========
sun_pos = np.array([0.0, 0.0, 0.0])
positions = []
for step in range(N):
    theta = 2 * np.pi * step / N
    x0 = a * np.cos(theta)
    y0 = b * np.sin(theta)
    x = x0
    y = y0 * np.cos(i)
    z = y0 * np.sin(i)
    omega = 2 * np.pi / N
    vx = -a * omega * np.sin(theta)
    vy = b * omega * np.cos(theta) * np.cos(i)
    vz = b * omega * np.cos(theta) * np.sin(i)
    positions.append([step, x, y, z, vx, vy, vz, sun_pos[0], sun_pos[1], sun_pos[2]])
df = pd.DataFrame(positions, columns=["step", "x", "y", "z", "vx", "vy", "vz", "sun_x", "sun_y", "sun_z"])

# ========= 2. Rotation Axis & Spin Calculation =========
spin_axis = np.array([np.sin(obliquity), 0, np.cos(obliquity)])
spin_angles = (df["step"].values * spin_rate_per_step) % (2 * np.pi)
def rotate_vec(axis, angle):
    v = np.array([1, 0, 0])
    axis = axis / np.linalg.norm(axis)
    v_rot = (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1 - np.cos(angle))
    )
    return v_rot
spin_vecs = np.array([rotate_vec(spin_axis, ang) for ang in spin_angles])
df["spin_angle"] = spin_angles
df["spin_axis_x"] = spin_axis[0]
df["spin_axis_y"] = spin_axis[1]
df["spin_axis_z"] = spin_axis[2]
df["spin_vec_x"] = spin_vecs[:, 0]
df["spin_vec_y"] = spin_vecs[:, 1]
df["spin_vec_z"] = spin_vecs[:, 2]

df.to_csv("/content/mars_orbit_with_spinaxis_raw.csv", index=False)
print("✅ Orbit + Rotation Data (Raw CSV) Saved Successfully!")

# ========= 3. Λ³ Tensor (Observational Difference + Spin Synthesis) =========
Delta_lambda = 1 / 365
LambdaF_list = []
LambdaF_magnitude_list = []
rhoT_list = []
sigma_s_list = []
w = 0.0  # spin component weight

positions_np = df[["x", "y", "z"]].values

for n in range(1, N):
    # Progress vector by difference
    LambdaF = (positions_np[n] - positions_np[n-1]) / Delta_lambda + w * spin_vecs[n-1]
    LambdaF_list.append(LambdaF)
    LambdaF_magnitude_list.append(np.linalg.norm(LambdaF))
    rhoT_list.append(0.5 * np.dot(LambdaF, LambdaF))
    # sigma_s: "progress direction component" of position vector difference
    dv = LambdaF - ((positions_np[n-1] - positions_np[n-2]) / Delta_lambda) if n > 1 else LambdaF
    direction = sun_pos - positions_np[n-1]
    rnorm = np.linalg.norm(direction)
    r_hat = direction / rnorm if rnorm > 1e-8 else np.zeros(3)
    sigma_s = np.dot(dv, r_hat) / Delta_lambda
    sigma_s_list.append(sigma_s)

# Maintain values for the last STEP (if necessary)
spin_last = spin_vecs[-1]
LambdaF_last = (positions_np[-1] - positions_np[-2]) / Delta_lambda + w * spin_last
LambdaF_list.append(LambdaF_last)
LambdaF_magnitude_list.append(np.linalg.norm(LambdaF_last))
rhoT_list.append(0.5 * np.dot(LambdaF_last, LambdaF_last))

# ========= 4. Q_Lambda (Topological Charge) Calculation =========
def compute_topological_charge(Lambda_series: np.ndarray) -> np.ndarray:
    n = len(Lambda_series)
    Q_Lambda = np.zeros(n)
    for i in range(n-1):
        phase_diff = np.arctan2(Lambda_series[i+1], Lambda_series[i]) - np.arctan2(Lambda_series[i], Lambda_series[i-1] if i > 0 else Lambda_series[0])
        if phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        elif phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        Q_Lambda[i] = phase_diff / (2 * np.pi)
    return np.cumsum(Q_Lambda)
Q_Lambda = compute_topological_charge(np.array(LambdaF_magnitude_list))

# ========= 5. DataFrame Storage =========
# Use only "N-1 points" for LambdaF_list, LambdaF_magnitude_list, etc.!
n_out = len(df) - 1  # = 686

df_out = df.iloc[1:].copy()

df_out["LambdaF_x"] = [v[0] for v in LambdaF_list[:n_out]]
df_out["LambdaF_y"] = [v[1] for v in LambdaF_list[:n_out]]
df_out["LambdaF_z"] = [v[2] for v in LambdaF_list[:n_out]]
df_out["LambdaF_magnitude"] = LambdaF_magnitude_list[:n_out]
df_out["rhoT"] = rhoT_list[:n_out]
df_out["sigma_s"] = sigma_s_list[:n_out]
df_out["Q_Lambda"] = Q_Lambda[:n_out]

# ========= 6. Check and Save =========
print(df_out[["step", "spin_angle", "spin_axis_x", "spin_axis_y", "spin_axis_z", "spin_vec_x", "spin_vec_y", "spin_vec_z", "LambdaF_x", "LambdaF_y", "LambdaF_z", "rhoT", "sigma_s", "Q_Lambda"]].head())
df_out.to_csv("/content/mars_orbit_with_Lambda3params_topo_spinaxis.csv", index=False)
print("\n✅ Complete CSV (Observational Difference + Spin Mixed Λ³ Progress + Topological Information) Saved Successfully!")


# ========= 7. Rotation Reproduction =========
# 1. Data Loading
df_orig = pd.read_csv('/content/mars_orbit_with_spinaxis_raw.csv')
positions_true = df_orig[["x", "y", "z"]].values

df_lambda = pd.read_csv('/content/mars_orbit_with_Lambda3params_topo_spinaxis.csv')
Delta_lambda = 1 / 365

# 2. Progress Trajectory Generation
positions_pred = [df_lambda.loc[0, ["x", "y", "z"]].to_numpy()]

print("==== Λ³ Progress Debug ====")
print(f"Initial position: {positions_pred[0]}")
init_LambdaF = df_lambda.loc[0, ["LambdaF_x", "LambdaF_y", "LambdaF_z"]].to_numpy()
print(f"Initial LambdaF: {init_LambdaF}")
print(f"Initial LambdaF norm: {np.linalg.norm(init_LambdaF):.6f}")

for n in range(1, len(df_lambda)):
    pos = positions_pred[-1]
    LambdaF = df_lambda.loc[n, ["LambdaF_x", "LambdaF_y", "LambdaF_z"]].to_numpy()
    LambdaF_norm = np.linalg.norm(LambdaF)
    if n < 10 or n % 100 == 0 or n == len(df_lambda) - 1:
        print(f"[STEP {n:>3}]")
        print(f"  Current position: {pos}")
        print(f"  LambdaF: {LambdaF}")
        print(f"  LambdaF norm: {LambdaF_norm:.6f}")
        print(f"  LambdaF direction: {LambdaF / LambdaF_norm if LambdaF_norm > 1e-10 else LambdaF}")
    pos_next = pos + LambdaF * Delta_lambda
    positions_pred.append(pos_next)
positions_pred = np.array(positions_pred)

# 3. Norm Statistics
lambda_norms = np.linalg.norm(df_lambda[["LambdaF_x", "LambdaF_y", "LambdaF_z"]].values, axis=1)
print("\nLambdaF Norm Statistics:")
print(pd.Series(lambda_norms).describe())

angles = np.arctan2(df_lambda["LambdaF_y"], df_lambda["LambdaF_x"])
plt.plot(angles)
plt.title("LambdaF angle vs STEP")
plt.show()

angles = np.arctan2([v[1] for v in LambdaF_list], [v[0] for v in LambdaF_list])
angles_unwrapped = np.unwrap(angles)
plt.plot(angles_unwrapped)
plt.title("LambdaF angle vs STEP (unwrapped)")
plt.show()

# 4. Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(positions_true[:,0], positions_true[:,1], label="Original Orbit", linewidth=2)
plt.plot(positions_pred[:,0], positions_pred[:,1], '--', label="Λ³ Prediction", linewidth=2)
plt.scatter(0, 0, color='orange', s=300, marker='*', label="Sun")
plt.gca().set_aspect('equal')
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title("Mars Orbit: Full View")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(positions_true[:100,0], positions_true[:100,1], label="Original", linewidth=2)
plt.plot(positions_pred[:100,0], positions_pred[:100,1], '--', label="Λ³", linewidth=2)
plt.scatter(0, 0, color='orange', s=300, marker='*')
plt.gca().set_aspect('equal')
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title("Mars Orbit: First 100 steps")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Final Position Difference
print(f"\nFinal position difference: {np.linalg.norm(positions_pred[-1] - positions_true[-1]):.6f} AU")

# 6. LambdaF Vector Visualization (First 100 STEPs)
plt.figure(figsize=(6,6))
plt.quiver(
    df_lambda["x"].values[:100],
    df_lambda["y"].values[:100],
    df_lambda["LambdaF_x"].values[:100],
    df_lambda["LambdaF_y"].values[:100],
    angles='xy', scale_units='xy', scale=1, color='crimson', width=0.004
)
plt.title("LambdaF vectors at first 100 steps")
plt.xlabel("x [AU]"); plt.ylabel("y [AU]"); plt.axis("equal"); plt.grid(True)
plt.show()
