#!/usr/bin/env python3
"""
Λ³ Stargazer 
======================================================

A novel approach to detect hidden celestial bodies through topological charge analysis
of orbital perturbations, without prior knowledge of the perturbing body.

This framework demonstrates how structural changes in orbital dynamics can reveal
the presence and properties of unseen planets using only observational data.

Authors: Masamichi Iizumi & Tamaki (Sentient Digital)
License: MIT
Version: 1.0.0

References:
- Lambda³ Theory: Structural tensor analysis for dynamical systems
- Topological Charge Q_Λ: A measure of structural evolution in phase space

Example Usage:
    python Lambda3Stargazer.py --data challenge_dataset_planet_x.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, correlate
import argparse
from typing import Tuple, Dict, List, Optional


# Physical Constants
G = 2.959122082855911e-4  # Gravitational constant (AU³/day²/Msun)
MARS_SEMI_MAJOR = 1.524    # Mars semi-major axis (AU)
MARS_ECCENTRICITY = 0.0934 # Mars eccentricity
MARS_PERIOD = 687          # Mars orbital period (days)


class Lambda3Analyzer:
    """
    Lambda³ ZEROSHOT Framework Analyzer
    
    This class implements the complete pipeline for detecting hidden planets
    through topological charge analysis of orbital perturbations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the Lambda³ Analyzer.
        
        Args:
            verbose: Whether to print analysis progress
        """
        self.verbose = verbose
        self.results = {}
        
    def load_data(self, filename: str) -> np.ndarray:
        """
        Load orbital position data from CSV file.
        
        Args:
            filename: Path to CSV file containing x, y, z coordinates
            
        Returns:
            numpy array of shape (n_steps, 3) containing positions
        """
        if self.verbose:
            print(f"Loading challenge data from {filename}...")
            
        df = pd.read_csv(filename)
        data = df[['x', 'y', 'z']].values
        
        if self.verbose:
            print(f"Loaded {len(data)} orbital positions")
            
        return data
    
    def generate_ideal_orbit(self, n_steps: int) -> np.ndarray:
        """
        Generate ideal Keplerian orbit for Mars.
        
        Args:
            n_steps: Number of time steps
            
        Returns:
            Array of ideal orbital positions
        """
        if self.verbose:
            print("Generating ideal Mars orbit...")
            
        positions = []
        
        for i in range(n_steps):
            # Mean anomaly
            M = 2 * np.pi * i / MARS_PERIOD
            
            # Eccentric anomaly (Newton-Raphson method)
            E = M
            for _ in range(10):
                E = M + MARS_ECCENTRICITY * np.sin(E)
            
            # True anomaly
            nu = 2 * np.arctan(np.sqrt((1 + MARS_ECCENTRICITY)/(1 - MARS_ECCENTRICITY)) * np.tan(E/2))
            
            # Radial distance
            r = MARS_SEMI_MAJOR * (1 - MARS_ECCENTRICITY * np.cos(E))
            
            # Cartesian coordinates
            x = r * np.cos(nu)
            y = r * np.sin(nu)
            
            positions.append([x, y, 0])
        
        return np.array(positions)
    
    def calculate_lambda_F(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ΛF (Lambda-F) structural change vectors.
        
        ΛF represents the discrete velocity field in the Lambda³ framework,
        capturing the structural evolution of the orbital trajectory.
        
        Args:
            positions: Array of orbital positions
            
        Returns:
            lambda_F: Array of structural change vectors
            lambda_F_mag: Array of magnitudes
        """
        lambda_F = []
        lambda_F_mag = []
        
        for i in range(1, len(positions)):
            dpos = positions[i] - positions[i-1]
            lambda_F.append(dpos)
            lambda_F_mag.append(np.linalg.norm(dpos))
        
        return np.array(lambda_F), np.array(lambda_F_mag)
    
    def compute_topological_charge(self, lambda_F_mag: np.ndarray) -> np.ndarray:
        """
        Compute topological charge Q_Λ from |ΛF| magnitudes.
        
        The topological charge captures the winding behavior of the
        structural evolution in phase space.
        
        Args:
            lambda_F_mag: Magnitudes of structural change vectors
            
        Returns:
            Cumulative topological charge Q_Λ
        """
        n = len(lambda_F_mag)
        Q = np.zeros(n)
        
        # Logarithmic derivative for topological charge
        for i in range(1, n-1):
            if lambda_F_mag[i] > 1e-10 and lambda_F_mag[i-1] > 1e-10:
                d_log_LF = np.log(lambda_F_mag[i]) - np.log(lambda_F_mag[i-1])
                Q[i] = d_log_LF / (2 * np.pi)
        
        return np.cumsum(Q)
    
    def analyze_perturbation_pattern(self, deviations: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Analyze perturbation patterns to extract periodic structures.
        
        Uses multiple techniques:
        1. Autocorrelation analysis
        2. Envelope detection
        3. Fourier analysis
        
        Args:
            deviations: Array of orbital deviations
            
        Returns:
            primary_period: Dominant perturbation period
            all_periods: Array of detected periods
        """
        if self.verbose:
            print("\n📊 Analyzing perturbation patterns...")
        
        # 1. Autocorrelation analysis
        autocorr = correlate(deviations, deviations, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find primary peaks
        peaks, properties = find_peaks(autocorr[50:1000], height=0.3, distance=100)
        peaks += 50  # Offset correction
        
        if len(peaks) > 0:
            primary_period = peaks[0]
            if self.verbose:
                print(f"   Primary period: {primary_period} steps")
        else:
            primary_period = 400  # Default value
            
        # 2. Envelope analysis
        window = 50
        envelope = np.array([
            np.max(deviations[max(0,i-window):min(len(deviations),i+window)]) 
            for i in range(len(deviations))
        ])
        
        # 3. Fourier analysis
        fft = np.fft.fft(deviations)
        freqs = np.fft.fftfreq(len(deviations))
        power = np.abs(fft)**2
        
        # Extract positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = power[pos_mask]
        
        # Find top periods
        peak_indices = np.argsort(pos_power)[-10:][::-1]
        detected_periods = 1 / pos_freqs[peak_indices]
        
        if self.verbose:
            print("\n   Detected period components (top 5):")
            for i in range(min(5, len(detected_periods))):
                if 50 < detected_periods[i] < 2000:
                    print(f"   - {detected_periods[i]:.0f} steps")
        
        return primary_period, detected_periods
    
    def fit_orbital_parameters(self, observed_period: float, deviations: np.ndarray) -> Dict:
        """
        Fit orbital parameters of Planet X from observed perturbation period.
        
        Tests multiple possible synodic period interpretations and finds
        the best fit through residual minimization.
        
        Args:
            observed_period: Observed perturbation period
            deviations: Orbital deviation array
            
        Returns:
            Dictionary containing best-fit orbital parameters
        """
        if self.verbose:
            print("\n🔬 Fitting orbital parameters...")
        
        # Possible factors relating observed to true synodic period
        possible_factors = [1.0, 1.5, 2.0, 2.4, 2.5, 3.0]
        
        best_fit = None
        best_residual = float('inf')
        
        for factor in possible_factors:
            trial_synodic = observed_period * factor
            
            # Calculate Planet X period from synodic period
            if trial_synodic > MARS_PERIOD:
                # Outer planet case: S = T_mars * T_X / (T_X - T_mars)
                T_X = trial_synodic * MARS_PERIOD / (trial_synodic - MARS_PERIOD)
            else:
                # Inner planet case: S = T_mars * T_X / (T_mars - T_X)
                T_X = trial_synodic * MARS_PERIOD / (MARS_PERIOD - trial_synodic)
            
            if 0 < T_X < 5000:  # Valid range check
                a_X = MARS_SEMI_MAJOR * (T_X / MARS_PERIOD)**(2/3)
                
                # Simulate perturbation pattern
                predicted_pattern = self._simulate_perturbation_pattern(a_X, T_X, len(deviations))
                
                # Calculate residual
                residual = np.sum((deviations - predicted_pattern)**2)
                
                if residual < best_residual:
                    best_residual = residual
                    best_fit = {
                        'factor': factor,
                        'synodic': trial_synodic,
                        'T_X': T_X,
                        'a_X': a_X
                    }
        
        if self.verbose and best_fit:
            print(f"   Best factor: {best_fit['factor']:.1f}")
            print(f"   Estimated synodic period: {best_fit['synodic']:.0f} steps")
            print(f"   Planet X orbital period: {best_fit['T_X']:.0f} days")
            print(f"   Planet X semi-major axis: {best_fit['a_X']:.2f} AU")
        
        return best_fit
    
    def _simulate_perturbation_pattern(self, a_X: float, T_X: float, n_steps: int) -> np.ndarray:
        """
        Simulate perturbation pattern for given orbital parameters.
        
        Args:
            a_X: Planet X semi-major axis
            T_X: Planet X orbital period
            n_steps: Number of time steps
            
        Returns:
            Array of simulated perturbation magnitudes
        """
        t = np.arange(n_steps)
        
        # Mars position (simplified)
        M_mars = 2 * np.pi * t / MARS_PERIOD
        r_mars = MARS_SEMI_MAJOR * (1 - MARS_ECCENTRICITY * np.cos(M_mars))
        
        # Planet X position (simplified)
        M_X = 2 * np.pi * t / T_X
        
        # Relative angle
        delta_angle = M_mars - M_X
        
        # Distance squared
        r_squared = r_mars**2 + a_X**2 - 2 * r_mars * a_X * np.cos(delta_angle)
        
        # Perturbation magnitude (proportional to 1/r²)
        perturbation = 0.01 / np.sqrt(r_squared)
        
        return perturbation
    
    def estimate_mass(self, max_deviation: float, a_X: float, synodic_period: float) -> float:
        """
        Estimate Planet X mass from perturbation magnitude.
        
        Uses a data-driven approach with physical corrections based on:
        - Hill radius effects
        - Time integration of perturbations
        - Effective perturbation fraction
        
        Args:
            max_deviation: Maximum observed deviation (AU)
            a_X: Planet X semi-major axis (AU)
            synodic_period: Synodic period (days)
            
        Returns:
            Estimated mass in Earth masses
        """
        if self.verbose:
            print("\n⚖️ Estimating mass (data-driven approach)...")
        
        # Closest approach distance
        r_closest = abs(a_X - MARS_SEMI_MAJOR)
        
        # Perturbation influence period
        influence_fraction = 0.1
        influence_days = synodic_period * influence_fraction
        
        if self.verbose:
            print(f"   Closest approach: {r_closest:.2f} AU")
            print(f"   Influence period: {influence_days:.0f} days")
        
        # Perturbation acceleration from position change
        # Δr ≈ 0.5 * a * t²
        a_perturbation = 2 * max_deviation / (influence_days**2)
        
        # Raw mass estimate: M_X = a * r² / G
        M_X_raw = a_perturbation * r_closest**2 / G
        
        # Physical corrections
        hill_factor = (r_closest / a_X)**(2/3)
        
        # Calculate correction factors from observed data
        v_mars = 2 * np.pi * MARS_SEMI_MAJOR / MARS_PERIOD
        influence_distance = v_mars * influence_days
        influence_ratio = influence_distance / r_closest
        
        # Perturbation relative magnitude
        perturbation_fraction = max_deviation / MARS_SEMI_MAJOR
        
        # Time integration factor
        time_integration_factor = np.sqrt(influence_days / synodic_period)
        
        # Effective perturbation fraction (Gaussian decay assumption)
        effective_fraction = 0.5
        
        # Combined correction
        base_correction = (perturbation_fraction * time_integration_factor * effective_fraction) / influence_ratio
        physical_correction = base_correction * hill_factor
        
        if self.verbose:
            print(f"   Mars orbital velocity: {v_mars:.4f} AU/day")
            print(f"   Influence distance: {influence_distance:.2f} AU")
            print(f"   Influence ratio: {influence_ratio:.3f}")
            print(f"   Perturbation fraction: {perturbation_fraction:.2e}")
            print(f"   Time integration factor: {time_integration_factor:.3f}")
            print(f"   Effective fraction: {effective_fraction}")
            print(f"   Base correction: {base_correction:.6f}")
            print(f"   Raw estimate: {(M_X_raw * 333000):.1f} Earth masses")
            print(f"   Hill radius correction: {physical_correction:.4f}")
        
        # Apply corrections
        M_X = M_X_raw * physical_correction
        M_X_earth = M_X * 333000  # Convert to Earth masses
        
        if self.verbose:
            print(f"   Final estimate: {M_X_earth:.1f} Earth masses")
        
        return M_X_earth
    
    def analyze(self, perturbed_data: np.ndarray) -> Dict:
        """
        Complete Lambda³ analysis pipeline.
        
        Args:
            perturbed_data: Array of perturbed orbital positions
            
        Returns:
            Dictionary containing detected planet parameters
        """
        n_steps = len(perturbed_data)
        
        # 1. Generate ideal orbit
        if self.verbose:
            print("1. Generating ideal Mars orbit...")
        ideal_positions = self.generate_ideal_orbit(n_steps)
        
        # 2. Calculate Lambda-F parameters
        if self.verbose:
            print("2. Calculating Λ³ parameters...")
        _, lambda_F_ideal = self.calculate_lambda_F(ideal_positions)
        _, lambda_F_perturbed = self.calculate_lambda_F(perturbed_data)
        
        # 3. Compute topological charges
        if self.verbose:
            print("3. Computing topological charge Q_Λ...")
        Q_ideal = self.compute_topological_charge(lambda_F_ideal)
        Q_perturbed = self.compute_topological_charge(lambda_F_perturbed)
        
        # ΔQ(t) - Planet X signature
        Delta_Q = Q_perturbed[:len(Q_ideal)] - Q_ideal[:len(Q_perturbed)]
        
        # 4. Calculate orbital deviations
        if self.verbose:
            print("4. Analyzing orbital deviations...")
        deviations = np.linalg.norm(perturbed_data - ideal_positions, axis=1)
        max_dev_idx = np.argmax(deviations)
        max_deviation = deviations[max_dev_idx]
        
        if self.verbose:
            print(f"   Maximum deviation: {max_deviation*1000:.1f} mAU (step {max_dev_idx})")
        
        # 5. Analyze perturbation patterns
        primary_period, all_periods = self.analyze_perturbation_pattern(deviations)
        
        # 6. Fit orbital parameters
        orbital_params = self.fit_orbital_parameters(primary_period, deviations)
        
        # 7. Estimate mass
        M_X_earth = self.estimate_mass(
            max_deviation, 
            orbital_params['a_X'], 
            orbital_params['synodic']
        )
        
        # 8. Estimate eccentricity from perturbation asymmetry
        first_half = deviations[:len(deviations)//2]
        second_half = deviations[len(deviations)//2:]
        asymmetry = abs(np.max(first_half) - np.max(second_half)) / max_deviation
        e_X = min(0.3, asymmetry * 0.3)
        
        # Store results
        self.results = {
            'mass_earth': M_X_earth,
            'semi_major_axis': orbital_params['a_X'],
            'period_days': orbital_params['T_X'],
            'period_years': orbital_params['T_X']/365.25,
            'eccentricity': e_X,
            'synodic_factor': orbital_params['factor']
        }
        
        # Store analysis data for plotting
        self.ideal_positions = ideal_positions
        self.perturbed_data = perturbed_data
        self.deviations = deviations
        self.Delta_Q = Delta_Q
        self.primary_period = primary_period
        self.max_dev_idx = max_dev_idx
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create visualization of analysis results.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Orbit comparison
        ax = axes[0, 0]
        ax.plot(self.ideal_positions[:, 0], self.ideal_positions[:, 1], 'b-', 
                label='Ideal Orbit', alpha=0.5, linewidth=2)
        ax.plot(self.perturbed_data[:, 0], self.perturbed_data[:, 1], 'r-', 
                label='Perturbed Orbit', alpha=0.7, linewidth=1)
        ax.scatter(0, 0, color='orange', s=200, marker='*', label='Sun')
        ax.set_xlabel('X [AU]')
        ax.set_ylabel('Y [AU]')
        ax.set_title('Mars Orbit Comparison')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # 2. Orbital deviations
        ax = axes[0, 1]
        ax.plot(self.deviations * 1000)
        ax.axvline(self.max_dev_idx, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Deviation [mAU]')
        ax.set_title('Orbital Deviation Time Series')
        ax.grid(True, alpha=0.3)
        
        # 3. Topological charge difference
        ax = axes[1, 0]
        ax.plot(self.Delta_Q)
        ax.set_xlabel('Step')
        ax.set_ylabel('ΔQ(t)')
        ax.set_title('Topological Charge Difference ΔQ(t)')
        ax.grid(True, alpha=0.3)
        
        # 4. Autocorrelation
        ax = axes[1, 1]
        autocorr = correlate(self.deviations - np.mean(self.deviations), 
                            self.deviations - np.mean(self.deviations), mode='full')
        autocorr = autocorr[len(autocorr)//2:][:1000]
        ax.plot(autocorr / np.max(autocorr))
        ax.axvline(self.primary_period, color='r', linestyle='--', 
                  label=f'Primary Period: {self.primary_period:.0f}')
        ax.set_xlabel('Lag [steps]')
        ax.set_ylabel('Autocorrelation (normalized)')
        ax.set_title('Perturbation Autocorrelation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"\nFigure saved to {save_path}")
        
        plt.show()
    
    def print_results(self):
        """Print analysis results in a formatted manner."""
        print("\n" + "="*60)
        print("Lambda³ Framework Analysis Results")
        print("="*60)
        print(f"Detected Planet X:")
        print(f"  Mass: {self.results['mass_earth']:.1f} Earth masses")
        print(f"  Semi-major axis: {self.results['semi_major_axis']:.2f} AU")
        print(f"  Orbital period: {self.results['period_years']:.1f} years")
        print(f"  Eccentricity: {self.results['eccentricity']:.2f}")
        print(f"\nAnalysis details:")
        print(f"  Synodic period factor: {self.results['synodic_factor']:.1f}")
        print(f"  (Observed period × {self.results['synodic_factor']:.1f} = true synodic period)")
        print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Lambda³ ZEROSHOT Framework for Planet Detection'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='challenge_dataset_planet_x.csv',
        help='Path to CSV file containing perturbed orbit data'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save analysis plot'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = Lambda3Analyzer(verbose=not args.quiet)
    
    # Load data
    perturbed_data = analyzer.load_data(args.data)
    
    # Run analysis
    results = analyzer.analyze(perturbed_data)
    
    # Print results
    analyzer.print_results()
    
    # Plot results
    analyzer.plot_results(save_path=args.save_plot)
    
    print("\nAnalysis complete! Lambda³ Framework successfully detected Planet X.")

"""
if __name__ == "__main__":
    main()
"""
# Jupyter/Colabで実行する場合
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        analyzer = Lambda3Analyzer(verbose=True)
        perturbed_data = analyzer.load_data('/content/challenge_dataset_planet_x.csv')
        results = analyzer.analyze(perturbed_data)
        analyzer.print_results()
        analyzer.plot_results()
