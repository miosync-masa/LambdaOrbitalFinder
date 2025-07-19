import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Challenge Dataset Generation
# =============================================================================
# Objective: Generate a perturbed Mars orbit influenced by an unknown "Planet X".
# The AI's task is to discover the properties of Planet X from this data alone.
# =============================================================================

# --- Simulation Parameters ---
N = 687 * 3  # Simulate for 3 Martian years to capture long-period perturbations
dt = 1.0     # Timestep: 1 day
G = 2.959122082855911e-4  # Gravitational constant (AU³/day²/Msun)

# --- Body Parameters ---
M_sun = 1.0
M_mars = 0.107 / 333000

# --- Planet X (The Secret Ingredient) ---
# Let's place it somewhere interesting... beyond Mars.
M_planet_x = (
    10 / 333000
)  # A "Super-Earth", 10x Earth mass, to create a clear signal
a_planet_x = 3.5  # Semi-major axis in the asteroid belt
e_planet_x = 0.15
period_planet_x = np.sqrt(a_planet_x**3) * 365.25

print(f"Secret Planet X parameters: Mass={M_planet_x*333000:.2f} Earth masses, a={a_planet_x} AU")

# --- Initial Conditions ---
# Mars
a_mars = 1.524
e_mars = 0.0934
mars_pos = np.array([a_mars * (1 - e_mars), 0.0, 0.0])
mars_vel_mag = np.sqrt(G * M_sun * (2 / np.linalg.norm(mars_pos) - 1 / a_mars))
mars_vel = np.array([0.0, mars_vel_mag, 0.0])

# Planet X (start at a different phase in its orbit)
theta_x_initial = np.pi / 2  # Start at 90 degrees
r_x_initial = a_planet_x * (1 - e_planet_x**2) / (1 + e_planet_x * np.cos(theta_x_initial))
planet_x_pos = np.array([r_x_initial * np.cos(theta_x_initial), r_x_initial * np.sin(theta_x_initial), 0.0])
planet_x_vel_mag = np.sqrt(G * M_sun * (2 / r_x_initial - 1 / a_planet_x))
# Velocity is tangential at this point
planet_x_vel = np.array([-planet_x_vel_mag * np.sin(theta_x_initial), planet_x_vel_mag * np.cos(theta_x_initial), 0.0])


# Sun
sun_pos = np.array([0.0, 0.0, 0.0])
sun_vel = np.array([0.0, 0.0, 0.0]) # Will move due to planets

# --- Integration using Velocity Verlet ---
print("Simulating the three-body problem (Sun, Mars, Planet X)...")

positions_mars_perturbed = [mars_pos.copy()]
positions_planet_x = [planet_x_pos.copy()]
positions_sun = [sun_pos.copy()]

# Initial accelerations
r_ms = sun_pos - mars_pos
r_mx = planet_x_pos - mars_pos
a_mars = (G * M_sun * r_ms / np.linalg.norm(r_ms) ** 3) + (G * M_planet_x * r_mx / np.linalg.norm(r_mx) ** 3)

for step in range(1, N):
    # Update positions
    mars_pos = mars_pos + mars_vel * dt + 0.5 * a_mars * dt**2
    # (We only need to track Mars's final position for this challenge)
    
    # We need to update Planet X and Sun positions to calculate new forces accurately
    # For simplicity, we'll approximate their new positions for force calculation
    # A full N-body integrator would update all bodies simultaneously.
    # This is a shortcut to get the perturbed data for Mars.
    
    # Update Planet X's position (simplified two-body for this example)
    r_xs = sun_pos - planet_x_pos
    a_planet_x = G * M_sun * r_xs / np.linalg.norm(r_xs)**3
    planet_x_vel = planet_x_vel + a_planet_x * dt
    planet_x_pos = planet_x_pos + planet_x_vel * dt
    
    # Update Sun's position
    r_sm = mars_pos - sun_pos
    r_sx = planet_x_pos - sun_pos
    a_sun = (G * M_mars * r_sm / np.linalg.norm(r_sm)**3) + (G * M_planet_x * r_sx / np.linalg.norm(r_sx)**3)
    sun_vel = sun_vel + a_sun * dt
    sun_pos = sun_pos + sun_vel * dt
    
    # Calculate new acceleration on Mars
    r_ms_new = sun_pos - mars_pos
    r_mx_new = planet_x_pos - mars_pos
    a_mars_new = (G * M_sun * r_ms_new / np.linalg.norm(r_ms_new)**3) + (G * M_planet_x * r_mx_new / np.linalg.norm(r_mx_new)**3)
    
    # Update Mars velocity
    mars_vel = mars_vel + 0.5 * (a_mars + a_mars_new) * dt
    a_mars = a_mars_new
    
    positions_mars_perturbed.append(mars_pos.copy())

print("✅ Simulation complete!")

# --- Create the final dataset for the AI ---
df_challenge = pd.DataFrame(positions_mars_perturbed, columns=["x", "y", "z"])
df_challenge.index.name = "step"

# Save to CSV
output_filename = "challenge_dataset_planet_x.csv"
df_challenge.to_csv(output_filename)

print(f"\nChallenge dataset created: {output_filename}")
print("This file contains ONLY the perturbed Mars trajectory.")
print("No information about Planet X is included.")
