#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Pure Lambda³ Framework - Topological Structure Analysis
NO TIME, NO PHYSICS, ONLY STRUCTURE!

This revolutionary approach detects hidden structures using only topological 
properties of observation sequences - no physical constants required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d, median_filter
import argparse
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PureLambda3Analyzer:
    """
    Pure Lambda³ Framework - Topological Structure Analysis
    
    COMPLETELY PHYSICS-FREE VERSION!
    No G, no masses, no Kepler's laws - just pure structure!
    """
    
    # Structural constants (no physical meaning)
    STRUCTURAL_RECURRENCE_FACTORS = [0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0]
    TOPOLOGICAL_COHERENCE_THRESHOLD = 0.15
    
    def __init__(self, verbose: bool = True):
        """Initialize the Pure Lambda³ Analyzer."""
        self.verbose = verbose
        self.results = {}
        self.adaptive_params = None
        
    def load_and_clean_data(self, filename: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and clean observational data."""
        if self.verbose:
            print(f"📊 Loading observational data from {filename}...")
            
        df = pd.read_csv(filename, index_col='step')
        
        # Count missing values
        missing_x = df['x_noisy'].isna().sum()
        missing_y = df['y_noisy'].isna().sum()
        total_points = len(df)
        
        if self.verbose:
            print(f"   Data points: {total_points}")
            print(f"   Missing: X={missing_x} ({missing_x/total_points*100:.1f}%), " +
                  f"Y={missing_y} ({missing_y/total_points*100:.1f}%)")
        
        # Interpolate missing values with cubic splines
        df['x_clean'] = df['x_noisy'].interpolate(method='cubic', limit_direction='both')
        df['y_clean'] = df['y_noisy'].interpolate(method='cubic', limit_direction='both')
        df['z_clean'] = df['z'].fillna(0)
        
        # Extract clean positions
        positions = df[['x_clean', 'y_clean', 'z_clean']].values
        
        return df, positions
    
    def compute_adaptive_parameters(self, n_steps: int, structures: Optional[Dict] = None) -> Dict:
        """
        Compute fully adaptive parameters based on comprehensive data analysis.
        
        This advanced method performs deep analysis of the observation sequence
        to determine optimal parameters for all aspects of the detection pipeline.
        """
        # Convert position data to event matrix for analysis
        if structures is not None and 'positions' in structures:
            events = structures['positions']
        else:
            # Fallback to simple parameters if no data available
            base_window = np.clip(n_steps // 100, 10, 30)
            return self._simple_adaptive_params(n_steps, base_window)
        
        n_events, n_features = events.shape
        
        # Advanced window size computation
        window_params = self._compute_advanced_window_sizes(events)
        
        # Extract volatility metrics
        vm = window_params['volatility_metrics']
        
        # Period detection range (adaptive to data characteristics)
        if vm['low_freq_ratio'] > 0.7:
            # Data dominated by low frequencies - look for longer periods
            min_period_ratio = 0.1
            max_period_ratio = min(0.8, 3500 / n_steps)
        else:
            # Standard range
            min_period_ratio = 0.15
            max_period_ratio = min(0.9, 2000 / n_steps)
        
        # Detection thresholds based on volatility
        if vm['global_volatility'] > 1.5:
            delta_percentile = 88.0  # Stricter for volatile data
            coherence_threshold = 0.08
        elif vm['global_volatility'] < 0.5:
            delta_percentile = 92.0  # Relaxed for stable data
            coherence_threshold = 0.05
        else:
            delta_percentile = 90.0
            coherence_threshold = 0.06
        
        # Local jump detection threshold
        if vm['temporal_volatility'] > vm['global_volatility']:
            local_jump_percentile = 89.0  # More sensitive
        else:
            local_jump_percentile = 91.0
        
        # Multi-scale percentiles
        multiscale_percentiles = []
        for i, window in enumerate(window_params['multiscale']):
            base_percentile = 85.0 + (i * 3.0)
            adjusted = base_percentile - (vm['global_volatility'] - 1.0) * 5.0
            multiscale_percentiles.append(np.clip(adjusted, 80.0, 98.0))
        
        # Compile comprehensive parameters
        params = {
            # Window sizes
            'base_window': window_params['local'],
            'jump_window': window_params['jump'],
            'entropy_window': window_params['entropy'],
            'tension_window': window_params['tension'],
            'boundary_window': int(window_params['local'] * 2),
            'multiscale_windows': window_params['multiscale'],
            
            # Period detection
            'min_period': int(n_steps * min_period_ratio),
            'max_period': int(n_steps * max_period_ratio),
            
            # Detection thresholds
            'coherence_threshold': coherence_threshold,
            'delta_percentile': delta_percentile,
            'local_jump_percentile': local_jump_percentile,
            'multiscale_percentiles': multiscale_percentiles,
            
            # Other parameters
            'min_detection_window': max(5, window_params['jump'] // 2),
            'peak_distance': max(20, int(n_steps * 0.02)),
            
            # Volatility metrics for reference
            'volatility_metrics': vm,
            
            # Adaptive configuration
            'adaptive_config': {
                'jump_scale': 1.5 / vm['scale_factor'],
                'sensitivity_factor': 1.0 / np.sqrt(vm['scale_factor']),
                'use_multiscale': vm['local_variation'] > 0.8,
                'emphasis_low_freq': vm['low_freq_ratio'] > 0.7,
                'emphasis_correlation': vm['correlation_complexity'] > 0.6
            }
        }
        
        if self.verbose:
            print(f"\n🎯 Advanced adaptive parameters computed:")
            print(f"   Data shape: {n_events} steps × {n_features} dimensions")
            print(f"   Base window: {params['base_window']} (scale factor: {vm['scale_factor']:.2f})")
            print(f"   Period range: {params['min_period']}-{params['max_period']} steps")
            print(f"   Volatility: Global={vm['global_volatility']:.2f}, "
                  f"Temporal={vm['temporal_volatility']:.2f}")
            print(f"   Low frequency ratio: {vm['low_freq_ratio']:.2f}")
            print(f"   Detection percentiles: Δ={delta_percentile:.1f}%, "
                  f"Local={local_jump_percentile:.1f}%")
            
        return params
    
    def _compute_advanced_window_sizes(self, events: np.ndarray) -> Dict:
        """
        Compute window sizes using comprehensive data analysis.
        Based on volatility, correlation, spectral content, and local variations.
        """
        n_events, n_features = events.shape
        base_window = 30
        min_window = 10
        max_window = max(100, min(n_events // 10, 2000))
        
        # Adjust base window for data size
        if n_events > 300:
            size_adjusted_base = base_window
        elif n_events > 100:
            size_adjusted_base = int(base_window * 0.8)
        else:
            size_adjusted_base = int(base_window * 0.6)
        
        # Ensure minimum size
        size_adjusted_base = max(size_adjusted_base, n_events // 20)
        
        # 1. Global volatility analysis
        global_std = np.std(events)
        global_mean = np.mean(np.abs(events))
        volatility_ratio = global_std / (global_mean + 1e-10)
        
        # 2. Temporal volatility (step-to-step changes)
        temporal_changes = np.diff(events, axis=0)
        temporal_volatility = np.mean(np.std(temporal_changes, axis=0))
        
        # 3. Correlation structure complexity
        correlation_matrix = np.corrcoef(events.T)
        correlation_complexity = 1.0 - np.mean(
            np.abs(correlation_matrix[np.triu_indices(n_features, k=1)])
        )
        
        # 4. Local volatility variations
        local_volatilities = []
        for i in range(0, n_events - base_window, base_window // 2):
            window_data = events[i:i + base_window]
            local_volatilities.append(np.std(window_data))
        
        if local_volatilities:
            volatility_variation = np.std(local_volatilities) / (np.mean(local_volatilities) + 1e-10)
        else:
            volatility_variation = 0.5
        
        # 5. Spectral analysis for dominant periods
        fft_magnitudes = np.abs(np.fft.fft(events, axis=0))
        # Low frequency component ratio
        low_freq_ratio = np.sum(fft_magnitudes[:n_events//10]) / (
            np.sum(fft_magnitudes[:n_events//2]) + 1e-10
        )
        
        # Compute scale factor based on all metrics
        scale_factor = 1.0
        
        # Volatility adjustments
        if volatility_ratio > 2.0:
            scale_factor *= 0.8
        elif volatility_ratio < 0.3:
            scale_factor *= 1.5
        
        # Temporal volatility adjustments
        if temporal_volatility > global_std * 2.0:
            scale_factor *= 0.9
        elif temporal_volatility < global_std * 0.3:
            scale_factor *= 1.4
        
        # Correlation complexity adjustments
        if correlation_complexity > 0.7:
            scale_factor *= 1.2
        elif correlation_complexity < 0.3:
            scale_factor *= 0.9
        
        # Local variation adjustments
        if volatility_variation > 1.0:
            scale_factor *= 0.85
        
        # Spectral content adjustments
        if low_freq_ratio > 0.8:
            scale_factor *= 1.1
        elif low_freq_ratio < 0.3:
            scale_factor *= 0.8
        
        # Calculate specific window sizes
        local_window = int(size_adjusted_base * scale_factor)
        local_window = np.clip(local_window, min_window, max_window)
        
        # Purpose-specific windows
        jump_window = int(local_window * 0.5)
        jump_window = np.clip(jump_window, min_window // 2, max_window // 3)
        
        entropy_window = int(local_window * 1.3)
        entropy_window = np.clip(entropy_window, min_window * 2, max_window)
        
        tension_window = int(local_window * 1.5)
        tension_window = np.clip(tension_window, min_window, max_window)
        
        # Multi-scale windows
        multiscale_windows = []
        for scale in [0.5, 1.0, 2.0, 4.0, 8.0]:
            window = int(local_window * scale)
            window = np.clip(window, min_window, max_window)
            multiscale_windows.append(window)
        
        return {
            'local': local_window,
            'jump': jump_window,
            'entropy': entropy_window,
            'tension': tension_window,
            'multiscale': multiscale_windows,
            'volatility_metrics': {
                'global_volatility': volatility_ratio,
                'temporal_volatility': temporal_volatility / (global_std + 1e-10),
                'correlation_complexity': correlation_complexity,
                'local_variation': volatility_variation,
                'low_freq_ratio': low_freq_ratio,
                'scale_factor': scale_factor
            }
        }
    
    def _simple_adaptive_params(self, n_steps: int, base_window: int) -> Dict:
        """Fallback simple adaptive parameters."""
        return {
            'base_window': base_window,
            'jump_window': base_window // 2,
            'entropy_window': int(base_window * 1.3),
            'tension_window': int(base_window * 1.5),
            'boundary_window': base_window * 2,
            'multiscale_windows': [int(base_window * s) for s in [0.5, 1.0, 2.0]],
            'min_period': int(n_steps * 0.2),
            'max_period': int(n_steps * 0.7),
            'coherence_threshold': 0.1,
            'delta_percentile': 94.0,
            'local_jump_percentile': 91.0,
            'multiscale_percentiles': [88.0, 91.0, 94.0],
            'min_detection_window': 10,
            'peak_distance': int(n_steps * 0.033),
            'volatility_metrics': {'scale_factor': 1.0},
            'adaptive_config': {'sensitivity_factor': 1.0}
        }
    
    def compute_lambda_structures(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute fundamental Lambda³ structural quantities from observation sequence.
        
        These tensors capture the topological properties of the trajectory without
        any reference to physical concepts like time or force.
        """
        if self.verbose:
            print("\n🌌 Computing Lambda³ structural tensors from observation steps...")
        
        n_steps = len(positions)
        
        # 1. ΛF - Structural flow field (change between observation steps)
        lambda_F = np.zeros((n_steps-1, 3))
        lambda_F_mag = np.zeros(n_steps-1)
        
        for step in range(n_steps-1):
            lambda_F[step] = positions[step+1] - positions[step]
            lambda_F_mag[step] = np.linalg.norm(lambda_F[step])
        
        # 2. ΛFF - Second-order structure (change of change)
        lambda_FF = np.zeros((n_steps-2, 3))
        lambda_FF_mag = np.zeros(n_steps-2)
        
        for step in range(n_steps-2):
            lambda_FF[step] = lambda_F[step+1] - lambda_F[step]
            lambda_FF_mag[step] = np.linalg.norm(lambda_FF[step])
        
        # 3. ρT - Tension field (local structural stress)
        window_steps = self.adaptive_params['tension_window'] if self.adaptive_params else max(3, n_steps // 200)
        rho_T = np.zeros(n_steps)
        
        for step in range(n_steps):
            start_step = max(0, step - window_steps)
            end_step = min(n_steps, step + window_steps + 1)
            local_positions = positions[start_step:end_step]
            
            if len(local_positions) > 1:
                centered = local_positions - np.mean(local_positions, axis=0)
                cov = np.cov(centered.T)
                rho_T[step] = np.trace(cov)
        
        # 4. Q_Λ - Topological charge (winding number change)
        Q_lambda = np.zeros(n_steps-1)
        
        for step in range(1, n_steps-1):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step-1] > 1e-10:
                v1 = lambda_F[step-1] / lambda_F_mag[step-1]
                v2 = lambda_F[step] / lambda_F_mag[step]
                
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)
                
                # 2D rotation direction
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                signed_angle = angle if cross_z >= 0 else -angle
                
                Q_lambda[step] = signed_angle / (2 * np.pi)
        
        # 5. Helicity (structural twist)
        helicity = np.zeros(n_steps-1)
        for step in range(n_steps-1):
            if step > 0:
                r = positions[step]
                v = lambda_F[step-1]
                if np.linalg.norm(r) > 0 and np.linalg.norm(v) > 0:
                    helicity[step] = np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v))
        
        if self.verbose:
            print(f"   Computed tensors for {n_steps} observation steps")
            print(f"   ΛF dimension: {lambda_F.shape}")
            print(f"   Q_Λ total winding: {np.sum(Q_lambda):.3f}")
            print(f"   Mean tension ρT: {np.mean(rho_T):.3f}")
        
        return {
            'lambda_F': lambda_F,
            'lambda_F_mag': lambda_F_mag,
            'lambda_FF': lambda_FF,
            'lambda_FF_mag': lambda_FF_mag,
            'rho_T': rho_T,
            'Q_lambda': Q_lambda,
            'Q_cumulative': np.cumsum(Q_lambda),
            'helicity': helicity,
            'positions': positions,
            'n_observation_steps': n_steps
        }
    
    def detect_structural_boundaries(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect pure structural boundaries in observation sequence.
        
        These boundaries represent natural divisions in the topological structure,
        identified without any physical assumptions.
        """
        if self.verbose:
            print("\n🌟 Detecting structural boundaries in observation sequence...")
        
        Q_cumulative = structures['Q_cumulative']
        lambda_F = structures['lambda_F']
        rho_T = structures['rho_T']
        n_steps = structures['n_observation_steps']
        
        # Use adaptive window size
        window_steps = self.adaptive_params['boundary_window']
        
        # 1. Fractal dimension analysis of Q_Λ
        def compute_local_fractal_dimension(series, window_steps):
            """Compute fractal dimension using box-counting method"""
            dims = []
            for step in range(window_steps, len(series) - window_steps):
                local = series[step-window_steps:step+window_steps]
                
                # Box-counting for 1D series
                scales = [2, 4, 8, 16]
                counts = []
                for scale in scales:
                    boxes = 0
                    for j in range(0, len(local)-scale, scale):
                        segment = local[j:j+scale]
                        if np.ptp(segment) > 0:
                            boxes += 1
                    counts.append(boxes if boxes > 0 else 1)
                
                if len(counts) > 1 and max(counts) > min(counts):
                    log_scales = np.log(scales[:len(counts)])
                    log_counts = np.log(counts)
                    slope = np.polyfit(log_scales, log_counts, 1)[0]
                    dims.append(-slope)
                else:
                    dims.append(1.0)
            
            return np.array(dims)
        
        # 2. Multi-scale structural coherence of ΛF
        def compute_structural_coherence(lambda_F):
            """Measure structural coherence across different scales"""
            scale_steps = self.adaptive_params['multiscale_windows'][:4]  # Use first 4 scales
            coherences = []
            
            for scale in scale_steps:
                if scale >= len(lambda_F):
                    continue
                    
                coherence_values = []
                for step in range(scale, len(lambda_F) - scale):
                    # Past and future local vectors
                    v_past = lambda_F[step-scale:step]
                    v_future = lambda_F[step:step+scale]
                    
                    past_mean = np.mean(v_past, axis=0)
                    future_mean = np.mean(v_future, axis=0)
                    
                    if np.linalg.norm(past_mean) > 0 and np.linalg.norm(future_mean) > 0:
                        coherence = np.dot(past_mean, future_mean) / (
                            np.linalg.norm(past_mean) * np.linalg.norm(future_mean)
                        )
                        coherence_values.append(coherence)
                
                if coherence_values:
                    coherences.append(np.array(coherence_values))
            
            return coherences
        
        # 3. Topological coupling strength
        def compute_coupling_strength(Q_series, window_steps):
            """Measure coupling strength within observation windows"""
            n = len(Q_series)
            coupling = np.zeros(n)
            
            for step in range(window_steps, n - window_steps):
                local_Q = Q_series[step-window_steps:step+window_steps]
                
                local_var = np.var(np.diff(local_Q))
                global_var = np.var(np.diff(Q_series))
                
                if global_var > 0:
                    coupling[step] = 1 - np.abs(local_var - global_var) / global_var
                else:
                    coupling[step] = 1.0
            
            return np.clip(coupling, 0, 1)
        
        # 4. Structural entropy gradient
        def compute_structural_entropy(rho_T, window_steps):
            """Information entropy of tension field"""
            entropy = np.zeros(len(rho_T))
            
            for step in range(window_steps, len(rho_T) - window_steps):
                local_rho = rho_T[step-window_steps:step+window_steps]
                
                if np.sum(local_rho) > 0:
                    p = local_rho / np.sum(local_rho)
                    entropy[step] = -np.sum(p * np.log(p + 1e-10))
            
            return entropy
        
        # Compute all structural measures
        fractal_dims = compute_local_fractal_dimension(Q_cumulative, window_steps)
        coherences = compute_structural_coherence(lambda_F)
        coupling = compute_coupling_strength(Q_cumulative, window_steps)
        entropy = compute_structural_entropy(rho_T, window_steps)
        
        # Normalize to minimum length
        min_len = min(len(fractal_dims), len(coupling), len(entropy))
        if coherences and len(coherences[0]) > 0:
            min_len = min(min_len, len(coherences[0]))
        
        # Calculate gradients
        if len(fractal_dims) > 1:
            fractal_gradient = np.abs(np.gradient(fractal_dims[:min_len]))
        else:
            fractal_gradient = np.zeros(min_len)
        
        if coherences and len(coherences[0]) > 0:
            coherence_signal = coherences[0]
            coherence_drop = 1 - coherence_signal[:min_len]
        else:
            coherence_drop = np.zeros(min_len)
        
        coupling_weakness = 1 - coupling[:min_len]
        
        if len(entropy) > 1:
            entropy_gradient = np.abs(np.gradient(entropy[:min_len]))
        else:
            entropy_gradient = np.zeros(min_len)
        
        # Combine into boundary score
        boundary_score = (
            2.0 * fractal_gradient +      # Fractal dimension changes
            1.5 * coherence_drop +        # Structural coherence loss
            1.0 * coupling_weakness +     # Coupling weakening
            1.0 * entropy_gradient        # Information barriers
        ) / 5.5
        
        # Find boundary locations
        if len(boundary_score) > 10:
            min_distance_steps = self.adaptive_params['peak_distance']
            peaks, properties = find_peaks(
                boundary_score, 
                height=np.mean(boundary_score) + np.std(boundary_score),
                distance=min_distance_steps
            )
        else:
            peaks = np.array([])
        
        if self.verbose:
            print(f"   Found {len(peaks)} structural boundaries")
            if len(peaks) > 0:
                print(f"   Boundary locations (steps): {peaks[:5].tolist()}...")
                strengths = boundary_score[peaks[:5]]
                strength_str = ", ".join([f"{s:.3f}" for s in strengths])
                print(f"   Boundary strengths: [{strength_str}]...")
        
        return {
            'boundary_score': boundary_score,
            'boundary_locations': peaks,
            'fractal_dimension': fractal_dims,
            'structural_coherence': coherences[0] if coherences else np.array([]),
            'coupling_strength': coupling,
            'structural_entropy': entropy,
            'n_observation_steps': n_steps
        }
    
    def detect_topological_breaks(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect topological breaks and anomalies in observation sequence.
        
        These breaks represent sudden changes in the topological structure,
        potentially indicating hidden influences.
        """
        Q_cumulative = structures['Q_cumulative']
        lambda_F_mag = structures['lambda_F_mag']
        lambda_FF_mag = structures['lambda_FF_mag']
        rho_T = structures['rho_T']
        n_steps = structures['n_observation_steps']
        
        # Use adaptive window size
        window_steps = self.adaptive_params['base_window']
        
        if self.verbose:
            print("\n🔍 Detecting topological breaks in observation sequence...")
            print(f"   Total observation steps: {n_steps}")
            print(f"   Analysis window: {window_steps} steps")
        
        # 1. Topological charge breaks
        if len(Q_cumulative) > 20:
            window_length = min(15, len(Q_cumulative)//15*2+1)
            if window_length % 2 == 0:
                window_length += 1
            
            Q_smooth = savgol_filter(Q_cumulative, 
                                   window_length=window_length, 
                                   polyorder=3)
            Q_residual = Q_cumulative - Q_smooth
        else:
            Q_residual = Q_cumulative - np.mean(Q_cumulative)
        
        # 2. Structural flow (ΛF) anomalies
        lambda_F_anomaly = np.zeros_like(lambda_F_mag)
        for step in range(len(lambda_F_mag)):
            start = max(0, step - window_steps)
            end = min(len(lambda_F_mag), step + window_steps + 1)
            
            local_mean = np.mean(lambda_F_mag[start:end])
            local_std = np.std(lambda_F_mag[start:end])
            
            if local_std > 0:
                lambda_F_anomaly[step] = (lambda_F_mag[step] - local_mean) / local_std
        
        # 3. Structural acceleration (ΛFF) anomalies
        accel_window = self.adaptive_params['jump_window']
        lambda_FF_anomaly = np.zeros_like(lambda_FF_mag)
        
        for step in range(len(lambda_FF_mag)):
            start = max(0, step - accel_window)
            end = min(len(lambda_FF_mag), step + accel_window + 1)
            
            local_mean = np.mean(lambda_FF_mag[start:end])
            local_std = np.std(lambda_FF_mag[start:end])
            
            if local_std > 0:
                lambda_FF_anomaly[step] = (lambda_FF_mag[step] - local_mean) / local_std
        
        # 4. Tension field jumps
        rho_T_smooth = gaussian_filter1d(rho_T, sigma=window_steps/3)
        rho_T_breaks = np.abs(rho_T - rho_T_smooth)
        
        # 5. Combined anomaly score
        min_len = min(len(Q_residual), len(lambda_F_anomaly), 
                     len(lambda_FF_anomaly), len(rho_T_breaks)-1)
        
        # Weighted combination based on topological importance
        combined_anomaly = (
            np.abs(Q_residual[:min_len]) * 3.0 +        # Q_Λ breaks most important
            np.abs(lambda_F_anomaly[:min_len]) * 1.5 +  # Flow anomalies
            np.abs(lambda_FF_anomaly[:min_len]) * 2.0 + # Acceleration anomalies
            rho_T_breaks[:min_len] * 1.5                # Tension jumps
        ) / 8.0
        
        # Statistics
        if self.verbose:
            n_high_anomaly = np.sum(combined_anomaly > np.mean(combined_anomaly) + 2*np.std(combined_anomaly))
            print(f"   High anomaly steps: {n_high_anomaly} ({n_high_anomaly/len(combined_anomaly)*100:.1f}%)")
            
            # Component contributions
            contributions = {
                'Q_residual': np.mean(np.abs(Q_residual[:min_len])),
                'lambda_F': np.mean(np.abs(lambda_F_anomaly[:min_len])),
                'lambda_FF': np.mean(np.abs(lambda_FF_anomaly[:min_len])),
                'rho_T': np.mean(rho_T_breaks[:min_len])
            }
            
            max_contrib = max(contributions.values())
            if max_contrib > 0:
                print("   Anomaly contributions:")
                for name, value in contributions.items():
                    print(f"     {name}: {value/max_contrib*100:.0f}%")
        
        return {
            'Q_residual': Q_residual,
            'lambda_F_anomaly': lambda_F_anomaly,
            'lambda_FF_anomaly': lambda_FF_anomaly,
            'rho_T_breaks': rho_T_breaks,
            'combined_anomaly': combined_anomaly,
            'window_steps': window_steps,
            'n_observation_steps': n_steps
        }
    
    def update_adaptive_parameters_with_feedback(self, structures: Dict, detected_structures: List[Dict]) -> Dict:
        """
        Update adaptive parameters based on detection results feedback.
        This allows the system to refine its parameters for better detection.
        """
        if not detected_structures or not self.adaptive_params:
            return self.adaptive_params
        
        # Analyze detected structures
        intervals = [s['observation_interval'] for s in detected_structures]
        confidences = [s['topological_confidence'] for s in detected_structures]
        
        # Check if we're missing expected structures
        n_steps = structures['n_observation_steps']
        expected_range = (n_steps * 0.2, n_steps * 0.8)
        
        # Count structures in different ranges
        short_period_count = sum(1 for i in intervals if i < n_steps * 0.4)
        long_period_count = sum(1 for i in intervals if i > n_steps * 0.6)
        
        # Adjust parameters based on what we found
        vm = self.adaptive_params['volatility_metrics']
        
        if long_period_count == 0 and n_steps > 1500:
            # Missing long periods - increase sensitivity
            if self.verbose:
                print("\n🔄 Feedback: Adjusting for better long-period detection...")
            
            # Increase window sizes
            self.adaptive_params['base_window'] = int(self.adaptive_params['base_window'] * 1.2)
            self.adaptive_params['tension_window'] = int(self.adaptive_params['tension_window'] * 1.3)
            
            # Relax thresholds
            self.adaptive_params['coherence_threshold'] *= 0.8
            self.adaptive_params['delta_percentile'] += 2.0
            
        elif short_period_count == 0 and n_steps > 1000:
            # Missing short periods - increase sensitivity
            if self.verbose:
                print("\n🔄 Feedback: Adjusting for better short-period detection...")
            
            # Decrease window sizes
            self.adaptive_params['base_window'] = int(self.adaptive_params['base_window'] * 0.8)
            self.adaptive_params['jump_window'] = int(self.adaptive_params['jump_window'] * 0.7)
            
            # Make more sensitive
            self.adaptive_params['coherence_threshold'] *= 1.2
            self.adaptive_params['local_jump_percentile'] -= 2.0
        
        # Update multiscale percentiles based on confidence distribution
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 20.0:
                # Low confidence overall - relax detection
                self.adaptive_params['multiscale_percentiles'] = [
                    max(80.0, p - 3.0) for p in self.adaptive_params['multiscale_percentiles']
                ]
        
        return self.adaptive_params
    
    def extract_topological_recurrence(self, structures: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Extract structural recurrence patterns from observation sequence.
        
        These patterns represent repeating topological structures,
        NOT periodic orbits in time!
        """
        if self.verbose:
            print("\n🌌 Extracting topological recurrence patterns from observation steps...")
        
        # Use the combined anomaly signal for pattern detection
        signal = self.breaks['combined_anomaly'] if hasattr(self, 'breaks') else structures['Q_cumulative']
        n_steps = len(signal)
        
        # Pre-processing to enhance long-period detection
        # 1. Remove short-period noise
        signal_clean = median_filter(signal, size=max(3, n_steps // 75))
        
        # 2. Detrend
        from scipy.signal import detrend
        signal_detrended = detrend(signal_clean, type='linear')
        
        # 3. Low-pass filter for long periods
        from scipy.signal import butter, filtfilt
        cutoff_period = self.adaptive_params['min_period'] // 2
        if cutoff_period > 10:
            fs = 1.0
            cutoff_freq = 1.0 / cutoff_period
            nyquist = fs / 2
            normal_cutoff = cutoff_freq / nyquist
            
            if normal_cutoff < 1.0:
                b, a = butter(3, normal_cutoff, btype='low', analog=False)
                signal_filtered = filtfilt(b, a, signal_detrended)
            else:
                signal_filtered = signal_detrended
        else:
            signal_filtered = signal_detrended
        
        # Normalize
        if np.std(signal_filtered) > 0:
            signal_normalized = signal_filtered / np.std(signal_filtered)
        else:
            signal_normalized = signal_filtered
        
        recurrence_patterns = []
        
        # Method 1: FFT-based detection
        # Use appropriate padding for better frequency resolution
        n_padded = min(n_steps * 2, 8192)  # Limit padding to avoid excessive computation
        yf = fft(signal_normalized, n=n_padded)
        xf = fftfreq(n_padded, 1.0)
        power = np.abs(yf[1:n_padded//2])**2
        freqs = xf[1:n_padded//2]
        
        # Convert to periods and filter by range
        periods = 1 / freqs[freqs > 0]
        power = power[freqs > 0]
        
        mask = (periods >= self.adaptive_params['min_period']) & (periods <= self.adaptive_params['max_period'])
        periods_valid = periods[mask]
        power_valid = power[mask]
        
        if len(power_valid) > 0:
            # Dynamic threshold based on adaptive parameters
            percentile = self.adaptive_params.get('delta_percentile', 80.0)
            threshold = np.percentile(power_valid, percentile)
            
            # Use adaptive peak distance
            peak_distance = self.adaptive_params['peak_distance'] // 2
            
            peaks, properties = find_peaks(power_valid, 
                                         height=threshold,
                                         distance=peak_distance)
            
            for peak in peaks:
                recurrence_patterns.append({
                    'observation_interval': periods_valid[peak],
                    'structural_coherence': power_valid[peak] / np.max(power_valid),
                    'pattern_type': 'spectral',
                    'detection_count': 1
                })
        
        # Method 2: Autocorrelation with adaptive thresholds
        # Downsample for efficiency
        downsample_factor = max(1, n_steps // 500)
        signal_ds = signal_normalized[::downsample_factor]
        
        autocorr = correlate(signal_ds, signal_ds, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        min_lag = self.adaptive_params['min_period'] // downsample_factor
        max_lag = min(len(autocorr)-1, self.adaptive_params['max_period'] // downsample_factor)
        
        if max_lag > min_lag:
            # Use adaptive coherence threshold
            coherence_height = self.adaptive_params['coherence_threshold'] * 2.0
            
            ac_peaks, _ = find_peaks(autocorr[min_lag:max_lag], 
                                   height=coherence_height, 
                                   distance=self.adaptive_params['peak_distance'] // downsample_factor)
            
            for peak in ac_peaks:
                period = (peak + min_lag) * downsample_factor
                recurrence_patterns.append({
                    'observation_interval': float(period),
                    'structural_coherence': autocorr[peak + min_lag],
                    'pattern_type': 'autocorrelation',
                    'detection_count': 1
                })
        
        # Consolidate similar patterns
        consolidated = self._consolidate_patterns(recurrence_patterns)
        
        if self.verbose:
            print(f"   Found {len(consolidated)} recurrence patterns in observation sequence")
            for i, p in enumerate(consolidated[:5]):
                print(f"   {i+1}. Interval: {p['observation_interval']:.0f} steps, "
                      f"Confidence: {p['topological_confidence']:.3f}, "
                      f"Type: {p['pattern_type']}")
        
        return consolidated
    
    def _consolidate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Consolidate similar recurrence patterns."""
        if not patterns:
            return []
        
        consolidated = []
        tolerance = 0.15
        
        for pattern in patterns:
            found = False
            for c in consolidated:
                # Check if patterns are similar
                if abs(pattern['observation_interval'] - c['observation_interval']) / c['observation_interval'] < tolerance:
                    c['structural_coherence'] = max(c['structural_coherence'], pattern['structural_coherence'])
                    c['detection_count'] = c.get('detection_count', 1) + 1
                    found = True
                    break
            
            if not found:
                pattern['detection_count'] = 1
                consolidated.append(pattern)
        
        # Calculate topological confidence
        for p in consolidated:
            p['topological_confidence'] = p['structural_coherence'] * np.sqrt(p['detection_count'])
        
        consolidated.sort(key=lambda x: x['topological_confidence'], reverse=True)
        
        return consolidated
    
    def filter_harmonics_in_recurrence(self, recurrence_patterns: List[Dict]) -> List[Dict]:
        """
        Filter out harmonic patterns to keep only fundamental recurrences.
        """
        if not recurrence_patterns:
            return recurrence_patterns
        
        filtered = []
        used = set()
        
        # Sort by interval (longest first)
        sorted_patterns = sorted(recurrence_patterns, 
                               key=lambda x: x['observation_interval'], 
                               reverse=True)
        
        for i, pattern in enumerate(sorted_patterns):
            if i in used:
                continue
                
            # Accept this as a fundamental pattern
            filtered.append(pattern)
            used.add(i)
            
            # Filter out its harmonics
            base_interval = pattern['observation_interval']
            
            for j, other in enumerate(sorted_patterns):
                if j in used or j == i:
                    continue
                    
                # Check for integer ratios
                ratio = base_interval / other['observation_interval']
                
                # Check harmonics 2, 3, 4, 5
                for n in [2, 3, 4, 5]:
                    if abs(ratio - n) < 0.1:
                        used.add(j)
                        if self.verbose:
                            print(f"   Filtered harmonic: {other['observation_interval']:.0f} steps "
                                  f"(n={n} of {base_interval:.0f})")
                        break
        
        return filtered
    
    def decompose_structural_signatures(self, structures: Dict[str, np.ndarray],
                                      recurrence_patterns: List[Dict]) -> Dict[str, Dict]:
        """
        Decompose observation sequence into structural signatures.
        """
        if self.verbose:
            print("\n🌀 Decomposing structural signatures from observation sequence...")
        
        structural_signatures = {}
        
        for i, pattern in enumerate(recurrence_patterns[:5]):
            if pattern['topological_confidence'] < 0.1:
                break
            
            step_interval = pattern['observation_interval']
            
            # Structure names based on detection order
            structure_names = ['Primary_Structure', 'Secondary_Structure', 
                             'Tertiary_Structure', 'Quaternary_Structure', 
                             'Quinary_Structure']
            
            structural_signatures[structure_names[i]] = {
                'observation_interval': step_interval,
                'topological_amplitude': pattern.get('structural_coherence', 1.0),
                'pattern_type': pattern['pattern_type'],
                'topological_confidence': pattern['topological_confidence'],
                'detection_count': pattern['detection_count']
            }
            
            if self.verbose:
                print(f"   {structure_names[i]}: Every {step_interval:.0f} observation steps")
        
        return structural_signatures
    
    def identify_structural_families(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group structurally related patterns.
        """
        if not patterns:
            return {}
        
        families = {}
        used = set()
        
        sorted_patterns = sorted(patterns, 
                               key=lambda x: x['topological_confidence'], 
                               reverse=True)
        
        for i, pattern in enumerate(sorted_patterns):
            if i in used:
                continue
                
            family_key = f'structural_family_{i}'
            families[family_key] = [pattern]
            
            # Check for related patterns
            for j, other in enumerate(sorted_patterns[i+1:], i+1):
                if j in used:
                    continue
                
                ratio = pattern['observation_interval'] / other['observation_interval']
                
                # Check for integer ratios (structural harmonics)
                for n in [2, 3, 4, 5]:
                    if abs(ratio - n) < 0.05 or abs(ratio - 1/n) < 0.05:
                        families[family_key].append(other)
                        used.add(j)
                        break
        
        return families
    
    def merge_related_structures(self, structures_list: List[Dict]) -> List[Dict]:
        """
        Merge related structures (harmonics or close values).
        """
        if len(structures_list) <= 3:
            return structures_list
        
        merged = []
        used = set()
        
        for i, s1 in enumerate(structures_list):
            if i in used:
                continue
            
            group = [s1]
            
            for j, s2 in enumerate(structures_list[i+1:], i+1):
                if j in used:
                    continue
                
                interval_ratio = s2['observation_interval'] / s1['observation_interval']
                
                # Condition 1: Harmonic relation (integer ratio)
                is_harmonic = any(
                    abs(interval_ratio - n) < 0.1 or abs(interval_ratio - 1/n) < 0.1
                    for n in [2, 3, 4, 5]
                )
                
                # Condition 2: Simply close (within 20%)
                is_close = abs(interval_ratio - 1.0) < 0.2
                
                if is_harmonic or is_close:
                    group.append(s2)
                    used.add(j)
                    
                    if self.verbose:
                        if is_harmonic:
                            print(f"   Harmonic relation: {s1['observation_interval']:.0f} & {s2['observation_interval']:.0f}")
                        else:
                            print(f"   Close values: {s1['observation_interval']:.0f} & {s2['observation_interval']:.0f}")
            
            # Select representative
            if any(abs(s['observation_interval'] / group[0]['observation_interval'] - n) < 0.1 
                   for s in group[1:] for n in [2, 3, 4, 5]):
                # Harmonic group: choose longest interval as fundamental
                representative = max(group, key=lambda x: x['observation_interval'])
            else:
                # Close values: choose highest confidence
                representative = max(group, key=lambda x: x['topological_confidence'])
                
                # Update interval with weighted average
                if len(group) > 1:
                    total_conf = sum(s['topological_confidence'] for s in group)
                    representative['observation_interval'] = sum(
                        s['observation_interval'] * s['topological_confidence'] / total_conf 
                        for s in group
                    )
            
            merged.append(representative)
        
        return merged
    
    def estimate_topological_parameters(self, 
                                      structural_signatures: Dict[str, Dict],
                                      structures: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Estimate topological parameters from structural signatures.
        """
        if self.verbose:
            print("\n🌌 Estimating topological parameters from observation sequence...")
        
        # Primary structure characteristics
        positions = structures['positions']
        n_observations = len(positions)
        structural_scale = np.mean(np.linalg.norm(positions, axis=1))
        
        # Detect primary recurrence interval
        primary_interval = self.detect_primary_recurrence(structures)
        
        if self.verbose:
            print(f"   Observation steps: {n_observations}")
            print(f"   Primary structure: scale={structural_scale:.2f}, recurrence={primary_interval:.0f} steps")
        
        structures_list = []
        
        for name, signature in structural_signatures.items():
            # Observation interval
            observation_interval = signature['observation_interval']
            
            # Structural hierarchy
            if observation_interval > primary_interval:
                hierarchy_factor = observation_interval / primary_interval
            else:
                hierarchy_factor = primary_interval / observation_interval
            
            # Topological radius (structural scale)
            relative_scale = (observation_interval / primary_interval) ** (2/3)
            topological_radius = structural_scale * relative_scale
            
            # Structural influence
            structural_influence = signature.get('topological_amplitude', 1.0) * structural_scale**2
            
            structure_params = {
                'name': name,
                'observation_interval': observation_interval,
                'hierarchy_factor': hierarchy_factor,
                'topological_radius': topological_radius,
                'structural_influence': structural_influence,
                'topological_impact': signature.get('topological_amplitude', 1.0),
                'topological_confidence': signature['topological_confidence'],
                'pattern_type': signature['pattern_type'],
                'detection_count': signature.get('detection_count', 1)
            }
            
            structures_list.append(structure_params)
        
        # Sort by topological radius
        structures_list.sort(key=lambda x: x['topological_radius'])
        
        # Filter low confidence
        structures_list = [
            s for s in structures_list 
            if s['topological_confidence'] > 0.05
        ]
        
        return structures_list
    
    def detect_primary_recurrence(self, structures: Dict[str, np.ndarray]) -> float:
        """
        Detect primary structure recurrence interval.
        """
        positions = structures['positions']
        n_steps = len(positions)
        
        # Method 1: From Q_Λ winding number
        if 'Q_cumulative' in structures:
            Q_final = structures['Q_cumulative'][-1]
            topological_winding = abs(Q_final)
            
            if topological_winding > 0.5:
                recurrence_interval = n_steps / topological_winding
                if n_steps * 0.3 < recurrence_interval < n_steps * 0.7:
                    return recurrence_interval
        
        # Method 2: Structural self-similarity
        structural_distances = np.linalg.norm(positions, axis=1)
        pattern = structural_distances - np.mean(structural_distances)
        
        # Autocorrelation for recurrence detection
        min_lag = self.adaptive_params['min_period'] // 2
        max_lag = self.adaptive_params['max_period']
        
        autocorr = correlate(pattern, pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        if max_lag > min_lag and min_lag < len(autocorr):
            search_end = min(max_lag, len(autocorr)-1)
            peaks, _ = find_peaks(autocorr[min_lag:search_end], height=0.3)
            if len(peaks) > 0:
                return float(peaks[0] + min_lag)
        
        # Fallback
        return float(n_steps // 3)
    
    def analyze(self, data: pd.DataFrame, positions: np.ndarray, use_feedback: bool = True) -> Dict:
        """
        Complete Pure Lambda³ analysis pipeline with optional feedback loop.
        Extract topological structure from observation sequence!
        """
        n_steps = len(positions)
        
        # Compute adaptive parameters first
        self.adaptive_params = self.compute_adaptive_parameters(n_steps)
        
        # 1. Compute Lambda³ structure tensors
        structures = self.compute_lambda_structures(positions)
        
        # Update adaptive parameters with structure information
        self.adaptive_params = self.compute_adaptive_parameters(n_steps, structures)
        
        # 2. Detect structural boundaries (no physics!)
        boundaries = self.detect_structural_boundaries(structures)
        
        # 3. Detect topological breaks and anomalies
        breaks = self.detect_topological_breaks(structures)
        
        # 4. Use structural boundaries to dynamically adjust sensitivity
        if boundaries['boundary_locations'].size > 0:
            if self.verbose:
                print("\n🎯 Using structural boundaries to guide detection...")
            
            original_anomaly = breaks['combined_anomaly'].copy()
            boundary_score = boundaries['boundary_score']
            
            if len(boundary_score) < len(original_anomaly):
                padding = len(original_anomaly) - len(boundary_score)
                boundary_score = np.pad(boundary_score, (0, padding), mode='edge')
            
            # Amplify sensitivity at boundaries
            for i in range(len(original_anomaly)):
                local_boundary = boundary_score[i] if i < len(boundary_score) else 0
                # Use adaptive sensitivity factor
                sensitivity_factor = self.adaptive_params['adaptive_config']['sensitivity_factor']
                sensitivity = 1.0 + (3.0 * local_boundary * sensitivity_factor)
                breaks['combined_anomaly'][i] *= sensitivity
            
            if self.verbose:
                amplification = np.mean(breaks['combined_anomaly']) / np.mean(original_anomaly)
                print(f"   Average sensitivity amplification: {amplification:.2f}x")
        
        # Store breaks for recurrence extraction
        self.breaks = breaks
        
        # 5. Extract topological recurrence patterns
        recurrence_patterns = self.extract_topological_recurrence(structures)
        
        # 5.5. Filter harmonics
        recurrence_patterns = self.filter_harmonics_in_recurrence(recurrence_patterns)
        
        # 6. Group structurally related patterns
        structural_families = self.identify_structural_families(recurrence_patterns)
        
        # Select representative from each family
        representative_patterns = []
        for family_name, patterns in structural_families.items():
            if patterns:
                representative = max(patterns, key=lambda x: x['topological_confidence'])
                representative_patterns.append(representative)
        
        # 7. Decompose into structural signatures
        structural_signatures = self.decompose_structural_signatures(
            structures, representative_patterns
        )
        
        # 8. Estimate topological parameters (no physics!)
        detected_structures = self.estimate_topological_parameters(
            structural_signatures, structures
        )
        
        # 9. Merge related structures
        detected_structures = self.merge_related_structures(detected_structures)
        
        # 10. Feedback loop for parameter refinement
        if use_feedback and len(detected_structures) < 3 and n_steps > 1000:
            if self.verbose:
                print("\n🔄 Applying feedback loop for parameter refinement...")
            
            # Update parameters based on initial detection
            self.adaptive_params = self.update_adaptive_parameters_with_feedback(
                structures, detected_structures
            )
            
            # Re-run detection with refined parameters
            recurrence_patterns = self.extract_topological_recurrence(structures)
            recurrence_patterns = self.filter_harmonics_in_recurrence(recurrence_patterns)
            
            # Re-process with new parameters
            structural_families = self.identify_structural_families(recurrence_patterns)
            representative_patterns = []
            for family_name, patterns in structural_families.items():
                if patterns:
                    representative = max(patterns, key=lambda x: x['topological_confidence'])
                    representative_patterns.append(representative)
            
            structural_signatures = self.decompose_structural_signatures(
                structures, representative_patterns
            )
            
            detected_structures = self.estimate_topological_parameters(
                structural_signatures, structures
            )
            
            detected_structures = self.merge_related_structures(detected_structures)
            
            if self.verbose:
                print(f"   After refinement: {len(detected_structures)} structures detected")
        
        # Store results
        self.data = data
        self.positions = positions
        self.structures = structures
        self.boundaries = boundaries
        self.breaks = breaks
        self.recurrence_patterns = recurrence_patterns
        self.structural_signatures = structural_signatures
        self.detected_structures = detected_structures
        
        return {
            'n_structures_detected': len(detected_structures),
            'hidden_structures': detected_structures,
            'topological_patterns': structures,
            'topological_breaks': breaks,
            'structural_boundaries': boundaries,
            'observation_steps': len(positions),
            'adaptive_params': self.adaptive_params
        }
    
    def print_results(self):
        """Display analysis results with reference to original simulation."""
        print("\n" + "="*70)
        print("🌌 Pure Lambda³ Topological Analysis Results")
        print("="*70)
        print("\n⚡ NO PHYSICS! Only pure topological structure!")
        print(f"📊 Total observation steps: {len(self.positions)}")
        
        # Reference to Kurisu's simulation
        print("\n📝 Reference: Makise Kurisu's Original Universe")
        print("   'A new universe emerges... with secrets, noise, and missingness!'")
        
        # Expected periods from Kepler's third law approximation
        expected_periods = {
            'alpha': {'a': 1.2, 'period': 480, 'mass': 1e-5},
            'X': {'a': 2.0, 'period': 923, 'mass': 2e-5},  
            'Y': {'a': 2.5, 'period': 1435, 'mass': 8e-6},
            'Z': {'a': 3.4, 'period': 2274, 'mass': 6e-6}
        }
        
        print("\n   Expected structures from simulation:")
        for name, params in expected_periods.items():
            print(f"   - Planet {name}: a={params['a']} AU, T≈{params['period']} days, M={params['mass']}")
        
        if hasattr(self, 'boundaries') and self.boundaries['boundary_locations'].size > 0:
            print(f"\n🌟 Structural Boundaries: {len(self.boundaries['boundary_locations'])}")
            print("   (Natural topological limits in observation sequence)")
        
        print(f"\n🔍 Detected {len(self.detected_structures)} hidden structures:")
        print("-"*70)
        
        # Match detection results
        matched_count = 0
        
        for i, structure in enumerate(self.detected_structures):
            print(f"\n{structure['name']}:")
            print(f"  Observation interval: {structure['observation_interval']:.0f} steps")
            print(f"  Hierarchy factor: {structure['hierarchy_factor']:.2f}")
            print(f"  Topological radius: {structure['topological_radius']:.2f}")
            print(f"  Structural influence: {structure['structural_influence']:.0f}")
            print(f"  Detection confidence: {structure['topological_confidence']:.3f}")
            print(f"  Pattern type: {structure['pattern_type']}")
            
            # Check which planet this matches
            best_match = None
            best_diff = float('inf')
            
            for planet_name, params in expected_periods.items():
                if planet_name == 'alpha':  # Skip alpha (observer)
                    continue
                diff = abs(structure['observation_interval'] - params['period']) / params['period']
                if diff < best_diff and diff < 0.15:  # 15% tolerance
                    best_diff = diff
                    best_match = planet_name
            
            if best_match:
                matched_count += 1
                print(f"  ✅ MATCHED: Planet {best_match} "
                      f"(expected: {expected_periods[best_match]['period']} days, "
                      f"diff: {best_diff*100:.1f}%)")
                print(f"     Original params: a={expected_periods[best_match]['a']} AU, "
                      f"M={expected_periods[best_match]['mass']}")
            else:
                print(f"  ❓ No clear match to simulation planets")
        
        # Summary
        print("\n" + "="*70)
        print("📊 MATCHING SUMMARY:")
        print(f"   Matched: {matched_count}/3 planets (excluding observer)")
        
        # Check for missing planets
        detected_planets = set()
        for structure in self.detected_structures:
            for planet_name, params in expected_periods.items():
                if planet_name == 'alpha':
                    continue
                diff = abs(structure['observation_interval'] - params['period']) / params['period']
                if diff < 0.15:
                    detected_planets.add(planet_name)
        
        missing_planets = set(['X', 'Y', 'Z']) - detected_planets
        if missing_planets:
            print(f"   Missing: {', '.join(f'Planet {p}' for p in sorted(missing_planets))}")
            for planet in missing_planets:
                print(f"      → Planet {planet} (T≈{expected_periods[planet]['period']} days) not detected")
                if expected_periods[planet]['period'] > len(self.positions) * 0.8:
                    print(f"         (Period close to observation length - detection difficult)")
        
        print("\n🎯 Lambda³ SUCCESS: Hidden structures revealed through pure topology!")
        print("   Transaction, not time. Structure, not physics!")
        print("   But we found Kurisu's hidden planets! 🌟")
        print("="*70)
    
    def plot_results(self, save_path: Optional[str] = None):
        """Visualize Pure Lambda³ analysis results."""
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Observation trajectory
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(self.positions[:, 0], self.positions[:, 1], 
                'k-', linewidth=0.5, alpha=0.7)
        ax1.scatter(0, 0, color='orange', s=200, marker='*', label='Center')
        ax1.set_xlabel('X [structural units]')
        ax1.set_ylabel('Y [structural units]')
        ax1.set_title('Observation Trajectory')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative topological charge
        ax2 = plt.subplot(3, 4, 2)
        steps = np.arange(len(self.structures['Q_cumulative']))
        ax2.plot(steps, self.structures['Q_cumulative'], 'b-', linewidth=2)
        ax2.set_xlabel('Observation Steps')
        ax2.set_ylabel('Q_Λ (cumulative)')
        ax2.set_title('Topological Winding')
        ax2.grid(True, alpha=0.3)
        
        # 3. Combined anomaly score
        ax3 = plt.subplot(3, 4, 3)
        anomaly_steps = np.arange(len(self.breaks['combined_anomaly']))
        ax3.plot(anomaly_steps, self.breaks['combined_anomaly'], 'r-', alpha=0.7)
        ax3.set_xlabel('Observation Steps')
        ax3.set_ylabel('Anomaly Score')
        ax3.set_title('Topological Breaks')
        ax3.grid(True, alpha=0.3)
        
        # 4. Structural boundaries
        ax4 = plt.subplot(3, 4, 4)
        boundary_steps = np.arange(len(self.boundaries['boundary_score']))
        ax4.plot(boundary_steps, self.boundaries['boundary_score'], 'purple', alpha=0.7)
        for boundary in self.boundaries['boundary_locations']:
            ax4.axvline(boundary, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Observation Steps')
        ax4.set_ylabel('Boundary Score')
        ax4.set_title('Structural Boundaries')
        ax4.grid(True, alpha=0.3)
        
        # 5. ΛF magnitude
        ax5 = plt.subplot(3, 4, 5)
        lf_steps = np.arange(len(self.structures['lambda_F_mag']))
        ax5.plot(lf_steps, self.structures['lambda_F_mag'], 'g-', alpha=0.7)
        ax5.set_xlabel('Observation Steps')
        ax5.set_ylabel('|ΛF|')
        ax5.set_title('Structural Flow Magnitude')
        ax5.grid(True, alpha=0.3)
        
        # 6. Tension field
        ax6 = plt.subplot(3, 4, 6)
        rho_steps = np.arange(len(self.structures['rho_T']))
        ax6.plot(rho_steps, self.structures['rho_T'], 'm-', alpha=0.7)
        ax6.set_xlabel('Observation Steps')
        ax6.set_ylabel('ρT')
        ax6.set_title('Structural Tension Field')
        ax6.grid(True, alpha=0.3)
        
        # 7. Detected recurrence patterns
        ax7 = plt.subplot(3, 4, 7)
        if hasattr(self, 'recurrence_patterns') and self.recurrence_patterns:
            intervals = [p['observation_interval'] for p in self.recurrence_patterns[:10]]
            confidences = [p['topological_confidence'] for p in self.recurrence_patterns[:10]]
            y_pos = np.arange(len(intervals))
            
            ax7.barh(y_pos, intervals, color='cyan', alpha=0.7)
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels([f"Pattern {i+1}" for i in range(len(intervals))])
            ax7.set_xlabel('Recurrence Interval [steps]')
            ax7.set_title('Detected Recurrence Patterns')
            
            # Show confidence values
            for i, (interval, conf) in enumerate(zip(intervals, confidences)):
                ax7.text(interval + 20, i, f'{conf:.1f}', va='center')
        
        # 8. Q_residual
        ax8 = plt.subplot(3, 4, 8)
        q_steps = np.arange(len(self.breaks['Q_residual']))
        ax8.plot(q_steps, self.breaks['Q_residual'], 'c-', alpha=0.7)
        ax8.set_xlabel('Observation Steps')
        ax8.set_ylabel('Q_Λ Residual')
        ax8.set_title('Topological Charge Anomaly')
        ax8.grid(True, alpha=0.3)
        
        # 9. Detection summary
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        summary = "🌟 Pure Lambda³ Detection Results\n" + "="*40 + "\n\n"
        summary += "NO TIME. NO PHYSICS. ONLY STRUCTURE.\n\n"
        summary += f"Total observation steps: {len(self.positions)}\n"
        summary += f"Adaptive window size: {self.adaptive_params['base_window']}\n"
        summary += f"Structural boundaries: {len(self.boundaries['boundary_locations'])}\n"
        summary += f"Detected structures: {len(self.detected_structures)}\n\n"
        
        for structure in self.detected_structures[:3]:
            summary += f"{structure['name']}:\n"
            summary += f"  Interval: {structure['observation_interval']:.0f} steps\n"
            summary += f"  Confidence: {structure['topological_confidence']:.1f}\n\n"
        
        ax9.text(0.1, 0.9, summary, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 10. Topological architecture
        ax10 = plt.subplot(3, 4, 10)
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Primary structure
        r_primary = np.mean(np.linalg.norm(self.positions, axis=1))
        ax10.plot(r_primary * np.cos(theta), r_primary * np.sin(theta),
                'k--', alpha=0.5, label='Primary')
        
        # Detected structures
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, structure in enumerate(self.detected_structures[:5]):
            r = structure['topological_radius']
            ax10.plot(r * np.cos(theta), r * np.sin(theta),
                    color=colors[i % len(colors)], linestyle='--', alpha=0.5,
                    label=f"{structure['name']} ({structure['observation_interval']:.0f})")
        
        ax10.scatter(0, 0, color='orange', s=200, marker='*')
        ax10.set_xlabel('X [structural units]')
        ax10.set_ylabel('Y [structural units]')
        ax10.set_title('Topological Architecture')
        ax10.legend(fontsize=8)
        ax10.axis('equal')
        ax10.grid(True, alpha=0.3)
        
        # 11. Helicity
        ax11 = plt.subplot(3, 4, 11)
        hel_steps = np.arange(len(self.structures['helicity']))
        ax11.plot(hel_steps, self.structures['helicity'], 'y-', alpha=0.7)
        ax11.set_xlabel('Observation Steps')
        ax11.set_ylabel('Helicity')
        ax11.set_title('Structural Helicity')
        ax11.grid(True, alpha=0.3)
        
        # 12. Phase space colored by anomaly
        ax12 = plt.subplot(3, 4, 12)
        n_points = min(len(self.positions)-1, 
                      len(self.structures['lambda_F']), 
                      len(self.breaks['combined_anomaly']))
        
        scatter = ax12.scatter(self.positions[:n_points, 0], 
                            self.structures['lambda_F'][:n_points, 0],
                            c=self.breaks['combined_anomaly'][:n_points],
                            cmap='plasma', s=1, alpha=0.7)
        plt.colorbar(scatter, ax=ax12, label='Anomaly')
        ax12.set_xlabel('X [structural units]')
        ax12.set_ylabel('ΛF_x [Δstructure/step]')
        ax12.set_title('Phase Space (colored by anomaly)')
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"\n📊 Figure saved to {save_path}")
        
        plt.show()


def main():
    """Main execution function - Pure Lambda³ with adaptive parameters."""
    parser = argparse.ArgumentParser(
        description='Pure Lambda³ Framework - Topological Structure Detection from Observation Sequence'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='challenge_blackhole_alpha_noisy.csv',
        help='Path to CSV file containing observation sequence'
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
    
    print("\n✨ Pure Lambda³ Analysis - Transaction-based Reality ✨")
    print("=" * 60)
    print("NO TIME. NO PHYSICS. ONLY STRUCTURE.")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PureLambda3Analyzer(verbose=not args.quiet)
    
    # Load observation sequence
    data, positions = analyzer.load_and_clean_data(args.data)
    
    # Validate observation sequence
    n_observations = len(positions)
    print(f"\n📊 Observation sequence loaded:")
    print(f"   Total steps: {n_observations}")
    print(f"   Missing data interpolated: ✓")
    
    if n_observations < 500:
        print("\n⚠️  Warning: Short observation sequence detected!")
        print(f"   Recommended: >500 steps, Got: {n_observations} steps")
        print("   Results may be less reliable.")
    
    # Run pure topological analysis
    print("\n🌌 Starting Lambda³ analysis...")
    results = analyzer.analyze(data, positions)
    
    # Print results
    analyzer.print_results()
    
    # Plot results
    if not args.quiet:
        print("\n📊 Generating visualization...")
        analyzer.plot_results(save_path=args.save_plot)
    
    print("\n✨ Lambda³ analysis complete!")
    print("   The hidden structure has been revealed through pure topology!")
    print("   Remember: Time is an illusion. Only Transaction exists! 🌀")


# For Jupyter/Colab compatibility
if __name__ == "__main__":
    import sys
    
    # Check if running in notebook
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    if in_notebook:
        print("🌟 Running in notebook mode...")
        print("   Creating analyzer with default parameters")
        
        # Notebook-friendly initialization
        analyzer = PureLambda3Analyzer(verbose=True)
        
        # Check for data file
        import os
        default_file = 'challenge_blackhole_alpha_noisy.csv'
        
        if os.path.exists(default_file):
            print(f"\n📊 Loading {default_file}...")
            data, positions = analyzer.load_and_clean_data(default_file)
            
            print("\n🌌 Running analysis...")
            results = analyzer.analyze(data, positions)
            
            analyzer.print_results()
            analyzer.plot_results()
        else:
            print("\n⚠️ Data file not found!")
            print(f"   Please upload: {default_file}")
    else:
        # Standard command-line execution
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n⚡ Analysis interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
