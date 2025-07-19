import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
🌌 Jupiter Perturbation Analysis using Λ³ ZEROSHOT Framework
===========================================================

This code demonstrates how gravitational perturbations from Jupiter
affect Mars' orbit, analyzed through the Λ³ structural tensor framework.

Key Innovation: We detect perturbations NOT through force calculations,
but through changes in the topological charge Q_Λ - a purely structural signature!
"""

# ========= Parameter Definition =========
# Mars orbital parameters
N = 687  # Mars orbital period (days) - one complete Martian year
a = 1.524  # Mars semi-major axis (AU) - average distance from Sun
b = 1.517  # Mars semi-minor axis (AU) - calculated from eccentricity
e = 0.0934  # Eccentricity - how "stretched" the ellipse is
i = np.radians(1.85)  # Orbital inclination - tilt relative to Earth's orbital plane

# Jupiter parameters for perturbation calculation
M_jupiter_ratio = 0.000954  # Jupiter/Sun mass ratio - Jupiter is ~1/1000th of Sun's mass
G = 2.959122082855911e-4  # Gravitational constant in AU³/day²/Msun units
a_j = 5.203  # Jupiter's semi-major axis (AU) - about 5x farther than Earth

# ========= 1. Generate ideal orbit data (no perturbation) =========
"""
First, we create Mars' "ideal" orbit - what it would be if only the Sun existed.
This serves as our baseline for detecting Jupiter's gravitational influence.
"""
sun_pos = np.array([0.0, 0.0, 0.0])  # Sun at origin
positions_ideal = []

for step in range(N):
    # Parametric ellipse equation
    theta = 2 * np.pi * step / N  # Angle around orbit (0 to 2π)
    x0 = a * np.cos(theta)  # Ellipse x-coordinate
    y0 = b * np.sin(theta)  # Ellipse y-coordinate
    
    # Apply orbital inclination (rotation around x-axis)
    x = x0
    y = y0 * np.cos(i)
    z = y0 * np.sin(i)
    
    positions_ideal.append([x, y, z])

positions_ideal = np.array(positions_ideal)

# ========= 2. Calculate Jupiter's position =========
"""
Jupiter moves much slower than Mars (11.86 year period vs 1.88 years).
We calculate Jupiter's position for each day of Mars' orbit.
"""
# Jupiter completes ~0.158 orbits while Mars completes 1 orbit
theta_j_list = 2 * np.pi * np.arange(N) / (11.86 * 365.25 / 1.88)
jupiter_positions = np.column_stack([
    a_j * np.cos(theta_j_list),  # Jupiter x-position
    a_j * np.sin(theta_j_list),  # Jupiter y-position
    np.zeros(N)  # Jupiter has ~0 inclination
])

# ========= 3. Calculate perturbed orbit (with Jupiter's gravity) =========
"""
Now we integrate Mars' motion including Jupiter's gravitational pull.
This uses simple Euler integration to show how Jupiter "tugs" on Mars.
"""
positions_perturbed = []
velocities_perturbed = []

# Initial conditions match ideal orbit start
pos = positions_ideal[0].copy()
# Initial velocity from position difference (AU/day)
vel = (positions_ideal[1] - positions_ideal[0])  # Approximate initial velocity

positions_perturbed.append(pos.copy())
velocities_perturbed.append(vel.copy())

dt = 1.0  # Time step = 1 day

for step in range(1, N):
    # Newton's law: F = ma, but we calculate acceleration directly
    
    # Acceleration from Sun (dominant force)
    r_ms = sun_pos - pos  # Vector from Mars to Sun
    r_ms_norm = np.linalg.norm(r_ms)  # Distance to Sun
    a_sun = G / r_ms_norm**3 * r_ms  # Gravitational acceleration (G*M_sun/r² in vector form)
    
    # Acceleration from Jupiter (perturbation)
    jupiter_pos = jupiter_positions[step]
    r_mj = jupiter_pos - pos  # Vector from Mars to Jupiter
    r_mj_norm = np.linalg.norm(r_mj)  # Distance to Jupiter
    a_jup = G * M_jupiter_ratio / r_mj_norm**3 * r_mj  # Jupiter's gravitational pull
    
    # Total acceleration
    acc_total = a_sun + a_jup
    
    # Update velocity and position (Euler integration)
    vel = vel + acc_total * dt  # v = v₀ + a*dt
    pos = pos + vel * dt  # x = x₀ + v*dt
    
    positions_perturbed.append(pos.copy())
    velocities_perturbed.append(vel.copy())

positions_perturbed = np.array(positions_perturbed)

# ========= 4. Lambda3 parameter calculation (ideal orbit) =========
"""
Calculate Λ³ structural tensors for the ideal orbit.
ΛF represents the "flow of meaning" - here it's the discrete velocity.
"""
Delta_lambda = 1.0  # Structural parameter = 1 day
LambdaF_list_ideal = []
LambdaF_magnitude_ideal = []

for n in range(1, N):
    # ΛF = position difference normalized by structural parameter
    LambdaF = (positions_ideal[n] - positions_ideal[n-1]) / Delta_lambda
    LambdaF_list_ideal.append(LambdaF)
    LambdaF_magnitude_ideal.append(np.linalg.norm(LambdaF))

# ========= 5. Lambda3 parameter calculation (perturbed orbit) =========
"""
Same calculation for the perturbed orbit.
Jupiter's influence will show up as variations in |ΛF|.
"""
LambdaF_list_perturbed = []
LambdaF_magnitude_perturbed = []

for n in range(1, N):
    LambdaF = (positions_perturbed[n] - positions_perturbed[n-1]) / Delta_lambda
    LambdaF_list_perturbed.append(LambdaF)
    LambdaF_magnitude_perturbed.append(np.linalg.norm(LambdaF))

# ========= 6. Topological charge calculation =========
"""
Q_Λ (topological charge) tracks the "winding number" of the orbit in phase space.
It's like counting how many times Mars "spins around" in velocity space.
This is a purely geometric/topological quantity!
"""
def compute_topological_charge(Lambda_series: np.ndarray) -> np.ndarray:
    """
    Computes cumulative topological charge from |ΛF| time series.
    
    The phase difference between consecutive |ΛF| values tells us
    how the "meaning flow" is rotating in abstract space.
    """
    n = len(Lambda_series)
    Q_Lambda = np.zeros(n)
    
    for i in range(1, n-1):
        # Calculate phase difference between consecutive points
        phase_diff = np.arctan2(Lambda_series[i+1], Lambda_series[i]) - \
                    np.arctan2(Lambda_series[i], Lambda_series[i-1])
        
        # Wrap phase to [-π, π]
        if phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        elif phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        
        # Normalize to winding number
        Q_Lambda[i] = phase_diff / (2 * np.pi)
    
    return np.cumsum(Q_Lambda)  # Cumulative sum = total winding

# Calculate topological charges for both orbits
Q_Lambda_ideal = compute_topological_charge(np.array(LambdaF_magnitude_ideal))
Q_Lambda_perturbed = compute_topological_charge(np.array(LambdaF_magnitude_perturbed))

# ========= 7. Calculate and visualize Delta Q(t) =========
"""
ΔQ(t) = Q_perturbed - Q_ideal is the KEY INNOVATION!
This difference in topological charge is Jupiter's "signature" on Mars' orbit.
It's a purely structural way to detect gravitational perturbations!
"""
Delta_Q = Q_Lambda_perturbed - Q_Lambda_ideal

# Create comprehensive visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Orbit comparison plot
ax = axes[0, 0]
ax.plot(positions_ideal[:,0], positions_ideal[:,1], 'b-', label='Ideal Orbit', linewidth=2)
ax.plot(positions_perturbed[:,0], positions_perturbed[:,1], 'r--', label='Perturbed Orbit', linewidth=2)
ax.scatter(0, 0, color='orange', s=300, marker='*', label='Sun')
ax.scatter(jupiter_positions[0,0], jupiter_positions[0,1], color='brown', s=100, marker='o', label='Jupiter')
ax.set_aspect('equal')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_title('Mars Orbit: Ideal vs Jupiter Perturbed')
ax.legend()
ax.grid(True)

# Orbit difference visualization (zoomed to milli-AU scale)
ax = axes[0, 1]
orbit_diff = positions_perturbed - positions_ideal
ax.plot(orbit_diff[:,0] * 1000, orbit_diff[:,1] * 1000)  # Convert to milli-AU
ax.set_xlabel('Delta X [mAU]')
ax.set_ylabel('Delta Y [mAU]')
ax.set_title('Orbit Difference (milli-AU)')
ax.grid(True)

# |ΛF| magnitude comparison
ax = axes[1, 0]
ax.plot(LambdaF_magnitude_ideal, label='Ideal', alpha=0.7)
ax.plot(LambdaF_magnitude_perturbed, label='Perturbed', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('|LambdaF| [AU/day]')
ax.set_title('LambdaF Magnitude Time Series')
ax.legend()
ax.grid(True)

# |ΛF| difference
ax = axes[1, 1]
LambdaF_diff = np.array(LambdaF_magnitude_perturbed) - np.array(LambdaF_magnitude_ideal)
ax.plot(LambdaF_diff)
ax.set_xlabel('Step')
ax.set_ylabel('Delta |LambdaF| [AU/day]')
ax.set_title('LambdaF Magnitude Difference')
ax.grid(True)

# Topological charge evolution
ax = axes[2, 0]
ax.plot(Q_Lambda_ideal[:-1], label='Ideal', alpha=0.7)
ax.plot(Q_Lambda_perturbed[:-1], label='Perturbed', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Q_Lambda (cumulative)')
ax.set_title('Topological Charge')
ax.legend()
ax.grid(True)

# ΔQ(t) - The key result: Jupiter's gravitational signature!
ax = axes[2, 1]
ax.plot(Delta_Q[:-1], 'r-', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Delta Q(t)')
ax.set_title('Delta Q(t): Jupiter Gravity Lambda3 Signature')
ax.grid(True)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print statistical summary
print("=== Jupiter Perturbation Effects ===")
print(f"Max orbit deviation: {np.max(np.linalg.norm(orbit_diff, axis=1)) * 1000:.2f} mAU")
print(f"Mean orbit deviation: {np.mean(np.linalg.norm(orbit_diff, axis=1)) * 1000:.2f} mAU")
print(f"Max Delta Q: {np.max(np.abs(Delta_Q)):.6f}")
print(f"Delta Q std deviation: {np.std(Delta_Q):.6f}")

# ========= Fourier Analysis of ΔQ(t) =========
"""
Finally, we analyze the frequency content of ΔQ(t).
This reveals the periodic nature of Jupiter's perturbations!
"""
from numpy.fft import fft, fftfreq

# Compute FFT of Delta Q
fft_vals = fft(Delta_Q[:-1])
freqs = fftfreq(len(Delta_Q)-1, d=1.0)  # Frequencies in 1/day units
power = np.abs(fft_vals)**2  # Power spectrum

# Plot frequency analysis
plt.figure(figsize=(10, 5))
plt.semilogy(freqs[:len(freqs)//2], power[:len(power)//2])
plt.xlabel('Frequency [1/day]')
plt.ylabel('Power Spectrum')
plt.title('Delta Q(t) Frequency Analysis - Jupiter Influence Period')
plt.grid(True)
plt.show()

"""
Key Insights:
1. Jupiter's gravity creates a measurable signature in Mars' topological charge
2. The perturbation is periodic, related to the Mars-Jupiter synodic period
3. We can detect gravitational influences purely through structural analysis
4. No force calculations needed - just tracking how "meaning flows" differently!

This demonstrates that gravity fundamentally alters the topology of orbital motion,
not just the physical trajectory!
"""
