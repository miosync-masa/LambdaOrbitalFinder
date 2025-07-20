import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# BLACK HOLE SYSTEM: Noisy & Missing Data Challenge
# =============================================================================
# Setup: Supermassive black hole (center), main planet α, three other planets (X, Y, Z)
# Goal: Can your model find the presence & effect of X, Y, Z from the noisy/incomplete α trajectory?
# =============================================================================

print("A new universe emerges... with secrets, noise, and missingness! -- by Makise Kurisu")

# --- Simulation Parameters ---
N = 1500  # ~4 years, more for long-term effect
dt = 1.0  # 1 day
G = 2.959122082855911e-4  # AU³/day²/Msun (works as generic constant)

# --- Body Parameters ---
M_bh = 1.0  # "Black hole" mass (arbitrary units)
M_alpha = 1e-5
M_X = 2e-5
M_Y = 8e-6
M_Z = 6e-6

# --- Orbital Elements ---
planets = {
    "alpha": dict(a=1.2, e=0.11, phi=0),
    "X":     dict(a=2.0, e=0.23, phi=2.0),
    "Y":     dict(a=2.5, e=0.14, phi=3.2),
    "Z":     dict(a=3.4, e=0.18, phi=5.8),
}
masses = {"alpha": M_alpha, "X": M_X, "Y": M_Y, "Z": M_Z}

# --- Initial conditions for each body ---
init_states = {}
for name, prm in planets.items():
    # Parametric ellipse at phi
    x = prm['a'] * (1 - prm['e']**2) / (1 + prm['e'] * np.cos(prm['phi']))
    y = 0
    pos = np.array([
        x * np.cos(prm['phi']),
        x * np.sin(prm['phi']),
        0.0
    ])
    # Vis-viva equation
    v_mag = np.sqrt(G * M_bh * (2 / np.linalg.norm(pos) - 1 / prm['a']))
    # Perpendicular velocity
    vx = -v_mag * np.sin(prm['phi'])
    vy = v_mag * np.cos(prm['phi'])
    vel = np.array([vx, vy, 0.0])
    init_states[name] = {"pos": pos, "vel": vel}

# --- Arrays to hold trajectories ---
positions = {name: [state["pos"].copy()] for name, state in init_states.items()}

# --- Velocity Verlet Integration (approximate) ---
for step in range(1, N):
    accs = {}
    # Calculate acceleration for each
    for name, state in init_states.items():
        pos = state["pos"]
        acc = -G * M_bh * pos / np.linalg.norm(pos) ** 3
        # Add mutual perturbations (main α gets all others, others only α)
        for other, om in masses.items():
            if name == other:
                continue
            pos_o = init_states[other]["pos"]
            acc += G * om * (pos_o - pos) / (np.linalg.norm(pos_o - pos) ** 3 + 1e-6)
        accs[name] = acc
    # Update positions/velocities
    for name in init_states:
        init_states[name]["vel"] += accs[name] * dt
        init_states[name]["pos"] += init_states[name]["vel"] * dt
        positions[name].append(init_states[name]["pos"].copy())

print("✅ Raw simulation complete!")

# --- Focus on 'alpha' trajectory only for output ---
alpha_traj = np.array(positions['alpha'])
df_challenge = pd.DataFrame(alpha_traj, columns=['x', 'y', 'z'])
df_challenge.index.name = 'step'

# =======================
# === NOISE GENERATION ==
# =======================

np.random.seed(42)
noise_std = 0.008  # Gaussian
jump_prob = 0.01   # 1% of steps: big jumps
jump_scale = 0.08

# --- Gaussian noise ---
df_challenge['x_noisy'] = df_challenge['x'] + np.random.normal(0, noise_std, N)
df_challenge['y_noisy'] = df_challenge['y'] + np.random.normal(0, noise_std, N)

# --- Random jumps ---
for col in ['x_noisy', 'y_noisy']:
    jumps = np.random.rand(N) < jump_prob
    df_challenge.loc[jumps, col] += np.random.normal(0, jump_scale, jumps.sum())

# --- Random missing data (simulate gaps) ---
missing_frac = 0.07  # 7% missing
for col in ['x_noisy', 'y_noisy']:
    missing_idx = np.random.choice(N, int(missing_frac * N), replace=False)
    df_challenge.loc[missing_idx, col] = np.nan

print(f"Noise (σ={noise_std}), jumps (p={jump_prob}, scale={jump_scale}), missing {missing_frac*100}% added!")

# --- Save and plot ---
filename = "challenge_blackhole_alpha_noisy.csv"
df_challenge[['x_noisy', 'y_noisy', 'z']].to_csv(filename)
print(f"\nChallenge file saved: {filename} (only alpha noisy trajectory)")

plt.figure(figsize=(8,8))
plt.plot(df_challenge['x'], df_challenge['y'], label="True α", alpha=0.3, color="blue")
plt.plot(df_challenge['x_noisy'], df_challenge['y_noisy'], '.', label="Noisy α obs", color="red", alpha=0.6)
plt.scatter(0, 0, color="black", marker="*", s=150, label="Black Hole (center)")
plt.title("Anomalous α Orbit (Noisy, Missing, Perturbed by X/Y/Z)")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.axis('equal')
plt.grid(True)
plt.legend()
