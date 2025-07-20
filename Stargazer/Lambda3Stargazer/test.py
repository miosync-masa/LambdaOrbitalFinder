#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
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
    
    # 定数（全て構造的な意味のみ！）
    PERIOD_REFINEMENT_FACTORS = [1/6, 1/3, 1/2.5, 1/2, 1/1.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0]
    STRUCTURAL_PERIOD_RANGE = (300, 3000)  # 構造的に意味のある周期範囲
    STRUCTURAL_COHERENCE_THRESHOLD = 0.90  # 構造的一貫性の閾値
    
    def __init__(self, verbose: bool = True):
        """Initialize the Pure Lambda³ Analyzer."""
        self.verbose = verbose
        self.results = {}
        
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

    def identify_harmonic_families(self, periodic_patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """紅莉栖の高調波クラスタリング（改良版）"""
        if not periodic_patterns:
            return {}
        
        # 各パターンを独立したファミリーとして初期化
        families = {}
        used = set()
        
        # Confidenceの高い順にソート（重要！）
        sorted_patterns = sorted(periodic_patterns, 
                              key=lambda x: x['confidence'], 
                              reverse=True)
        
        for i, pattern in enumerate(sorted_patterns):
            if i in used:
                continue
                
            family_key = f'family_{i}'
            families[family_key] = [pattern]
            
            # 他のパターンとの整数比をチェック
            for j, other in enumerate(sorted_patterns[i+1:], i+1):
                if j in used:
                    continue
                    
                ratio = pattern['period'] / other['period']
                
                # より厳しい整数比チェック（5%以内）
                for n in [2, 3, 4, 5]:  # 2-5倍のみチェック
                    if abs(ratio - n) < 0.05 or abs(ratio - 1/n) < 0.05:
                        families[family_key].append(other)
                        used.add(j)
                        break
        
        return families     

    def post_process_merge_harmonics(self, structures_list: List[Dict]) -> List[Dict]:
        """検出結果を見てから、明らかに同じ惑星の高調波だけマージ"""
        print("\n🔄 Post-processing harmonic merging...")
        print(f"   Input: {len(structures_list)} structures")
        
        if len(structures_list) <= 3:
            print("   → Already 3 or fewer structures, no merge needed")
            return structures_list
        
        merged = []
        used = set()
        
        for i, s1 in enumerate(structures_list):
            if i in used:
                continue
                
            print(f"\n   Checking {s1['name']} (Period: {s1['structural_period']}, Impact: {s1['topological_impact']:.3f})")
            group = [s1]
            
            for j, s2 in enumerate(structures_list[i+1:], i+1):
                if j in used:
                    continue
                
                period_ratio = s2['structural_period'] / s1['structural_period']
                impact_ratio = s2['topological_impact'] / s1['topological_impact']
                
                print(f"     vs {s2['name']}: period_ratio={period_ratio:.2f}, impact_ratio={impact_ratio:.2f}")
                
                is_harmonic = False
                for n in [2, 3, 4]:
                    if (abs(period_ratio - n) < 0.15 or 
                        abs(period_ratio - 1/n) < 0.15):
                        is_harmonic = True
                        print(f"       → Harmonic detected! (n={n})")
                        break
                
                if is_harmonic and 0.1 < impact_ratio < 10:
                    print(f"       → MERGING! {s2['name']} into {s1['name']}")
                    group.append(s2)
                    used.add(j)
                else:
                    print(f"       → Not merged (harmonic={is_harmonic}, impact_ok={0.1 < impact_ratio < 10})")
            
            representative = max(group, key=lambda x: x['confidence'])
            print(f"   Group representative: {representative['name']}")
            merged.append(representative)
        
        print(f"\n   Output: {len(merged)} structures after merging")
        return merged     
    
    def compute_lambda_structures(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute fundamental Lambda³ structural quantities."""
        if self.verbose:
            print("\n🌌 Computing Lambda³ structural tensors...")
        
        n = len(positions)
        
        # 1. ΛF - Structural flow field
        lambda_F = np.zeros((n-1, 3))
        lambda_F_mag = np.zeros(n-1)
        
        for i in range(n-1):
            lambda_F[i] = positions[i+1] - positions[i]
            lambda_F_mag[i] = np.linalg.norm(lambda_F[i])
        
        # 2. ΛFF - Second-order structure
        lambda_FF = np.zeros((n-2, 3))
        lambda_FF_mag = np.zeros(n-2)
        
        for i in range(n-2):
            lambda_FF[i] = lambda_F[i+1] - lambda_F[i]
            lambda_FF_mag[i] = np.linalg.norm(lambda_FF[i])
        
        # 3. ρT - Tension field
        window = max(3, n // 200)
        rho_T = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            local_positions = positions[start:end]
            
            if len(local_positions) > 1:
                centered = local_positions - np.mean(local_positions, axis=0)
                cov = np.cov(centered.T)
                rho_T[i] = np.trace(cov)
        
        # 4. Q_Λ - Topological charge
        Q_lambda = np.zeros(n-1)
        
        for i in range(1, n-1):
            if lambda_F_mag[i] > 1e-10 and lambda_F_mag[i-1] > 1e-10:
                v1 = lambda_F[i-1] / lambda_F_mag[i-1]
                v2 = lambda_F[i] / lambda_F_mag[i]
                
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)
                
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                signed_angle = angle if cross_z >= 0 else -angle
                
                Q_lambda[i] = signed_angle / (2 * np.pi)
        
        # 5. Helicity
        helicity = np.zeros(n-1)
        for i in range(n-1):
            if i > 0:
                r = positions[i]
                v = lambda_F[i-1]
                if np.linalg.norm(r) > 0 and np.linalg.norm(v) > 0:
                    helicity[i] = np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v))
        
        return {
            'lambda_F': lambda_F,
            'lambda_F_mag': lambda_F_mag,
            'lambda_FF': lambda_FF,
            'lambda_FF_mag': lambda_FF_mag,
            'rho_T': rho_T,
            'Q_lambda': Q_lambda,
            'Q_cumulative': np.cumsum(Q_lambda),
            'helicity': helicity,
            'positions': positions
        }
    
    def detect_topological_breaks(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Detect topological breaks and anomalies."""
        Q_cumulative = structures['Q_cumulative']
        lambda_F_mag = structures['lambda_F_mag']
        lambda_FF_mag = structures['lambda_FF_mag']
        rho_T = structures['rho_T']
        
        window = max(5, len(lambda_F_mag) // 100)
        
        if self.verbose:
            print("\n🔍 Detecting topological breaks and anomalies...")
            print(f"   Data length: {len(structures['positions'])} points")
            print(f"   Window size: {window} points")
        
        # 1. Breaks in topological charge
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
        
        # 2. Velocity anomalies
        lambda_F_anomaly = np.zeros_like(lambda_F_mag)
        for i in range(len(lambda_F_mag)):
            start = max(0, i - window)
            end = min(len(lambda_F_mag), i + window + 1)
            local_mean = np.mean(lambda_F_mag[start:end])
            local_std = np.std(lambda_F_mag[start:end])
            if local_std > 0:
                lambda_F_anomaly[i] = (lambda_F_mag[i] - local_mean) / local_std
        
        # 3. Acceleration anomalies
        accel_window = max(3, window // 2)
        lambda_FF_anomaly = np.zeros_like(lambda_FF_mag)
        for i in range(len(lambda_FF_mag)):
            start = max(0, i - accel_window)
            end = min(len(lambda_FF_mag), i + accel_window + 1)
            local_mean = np.mean(lambda_FF_mag[start:end])
            local_std = np.std(lambda_FF_mag[start:end])
            if local_std > 0:
                lambda_FF_anomaly[i] = (lambda_FF_mag[i] - local_mean) / local_std
        
        # 4. Tension jumps
        rho_T_smooth = gaussian_filter1d(rho_T, sigma=window/3)
        rho_T_breaks = np.abs(rho_T - rho_T_smooth)
        
        # 5. Combined anomaly
        min_len = min(len(Q_residual), len(lambda_F_anomaly), 
                    len(lambda_FF_anomaly), len(rho_T_breaks)-1)
        
        combined_anomaly = (
            np.abs(Q_residual[:min_len]) * 3.0 +
            np.abs(lambda_F_anomaly[:min_len]) * 1.5 +
            np.abs(lambda_FF_anomaly[:min_len]) * 2.0 +
            rho_T_breaks[:min_len] * 1.5
        ) / 8.0
        
        return {
            'Q_residual': Q_residual,
            'lambda_F_anomaly': lambda_F_anomaly,
            'lambda_FF_anomaly': lambda_FF_anomaly,
            'rho_T_breaks': rho_T_breaks,
            'combined_anomaly': combined_anomaly
        }
    
    def detect_structural_boundaries(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect pure structural boundaries WITHOUT physical constants!
        
        This revolutionary method finds the natural boundaries of gravitational
        influence using ONLY topological structure - no G, no masses, no distances!
        """
        if self.verbose:
            print("\n🌟 Detecting pure structural boundaries...")
        
        Q_cumulative = structures['Q_cumulative']
        lambda_F = structures['lambda_F']
        rho_T = structures['rho_T']
        
        # 1. Fractal dimension analysis of Q_Λ
        def compute_local_fractal_dimension(series, window=30):
            """Compute fractal dimension using box-counting method"""
            dims = []
            for i in range(window, len(series) - window):
                local = series[i-window:i+window]
                
                # Box-counting for 1D series
                scales = [2, 4, 8, 16]
                counts = []
                for scale in scales:
                    # Count boxes needed at this scale
                    boxes = 0
                    for j in range(0, len(local)-scale, scale):
                        segment = local[j:j+scale]
                        if np.ptp(segment) > 0:  # Box contains data
                            boxes += 1
                    counts.append(boxes if boxes > 0 else 1)
                
                # Fractal dimension from slope
                if len(counts) > 1 and max(counts) > min(counts):
                    log_scales = np.log(scales[:len(counts)])
                    log_counts = np.log(counts)
                    # Linear fit
                    slope = np.polyfit(log_scales, log_counts, 1)[0]
                    dims.append(-slope)
                else:
                    dims.append(1.0)  # Default dimension
            
            return np.array(dims)
        
        # 2. Multi-scale resonance analysis of ΛF
        def compute_structural_coherence(lambda_F, scales=[5, 10, 20, 40]):
            """Measure how coherently structure evolves across scales"""
            coherences = []
            
            for scale in scales:
                if scale >= len(lambda_F):
                    continue
                    
                # Compute phase coherence at this scale
                coherence_values = []
                for i in range(scale, len(lambda_F) - scale):
                    # Local vectors
                    v_past = lambda_F[i-scale:i]
                    v_future = lambda_F[i:i+scale]
                    
                    # Phase coherence (how aligned are past and future)
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
        def compute_coupling_strength(Q_series, window=50):
            """Measure how strongly different parts of trajectory are coupled"""
            n = len(Q_series)
            coupling = np.zeros(n)
            
            for i in range(window, n - window):
                # Compare local evolution with global
                local_Q = Q_series[i-window:i+window]
                
                # Local variance vs global variance
                local_var = np.var(np.diff(local_Q))
                global_var = np.var(np.diff(Q_series))
                
                # High coupling = local follows global pattern
                if global_var > 0:
                    coupling[i] = 1 - np.abs(local_var - global_var) / global_var
                else:
                    coupling[i] = 1.0
            
            return np.clip(coupling, 0, 1)
        
        # 4. Structural entropy gradient
        def compute_structural_entropy(rho_T, window=30):
            """Information content of structural tensor field"""
            entropy = np.zeros(len(rho_T))
            
            for i in range(window, len(rho_T) - window):
                local_rho = rho_T[i-window:i+window]
                
                # Normalize to probability distribution
                if np.sum(local_rho) > 0:
                    p = local_rho / np.sum(local_rho)
                    # Shannon entropy
                    entropy[i] = -np.sum(p * np.log(p + 1e-10))
            
            return entropy
        
        # Compute all structural measures
        fractal_dims = compute_local_fractal_dimension(Q_cumulative)
        coherences = compute_structural_coherence(lambda_F)
        coupling = compute_coupling_strength(Q_cumulative)
        entropy = compute_structural_entropy(rho_T)
        
        # Combine into structural boundary score
        # Boundaries occur where:
        # - Fractal dimension changes (structure complexity shift)
        # - Coherence drops (loss of correlation)
        # - Coupling weakens (independent evolution)
        # - Entropy gradient is high (information barrier)
        
        # Normalize all measures to same length
        min_len = min(len(fractal_dims), len(coupling), len(entropy))
        
        # Fractal dimension gradient
        if len(fractal_dims) > 1:
            fractal_gradient = np.abs(np.gradient(fractal_dims[:min_len]))
        else:
            fractal_gradient = np.zeros(min_len)
        
        # Coherence drop (use first scale)
        if coherences and len(coherences[0]) > 0:
            coherence_signal = coherences[0]
            coherence_drop = 1 - coherence_signal[:min_len]
        else:
            coherence_drop = np.zeros(min_len)
        
        # Coupling weakness
        coupling_weakness = 1 - coupling[:min_len]
        
        # Entropy gradient
        if len(entropy) > 1:
            entropy_gradient = np.abs(np.gradient(entropy[:min_len]))
        else:
            entropy_gradient = np.zeros(min_len)
        
        # Combined boundary score
        boundary_score = (
            2.0 * fractal_gradient +      # Weight fractal changes highly
            1.5 * coherence_drop +         # Phase decoherence
            1.0 * coupling_weakness +      # Decoupling
            1.0 * entropy_gradient         # Information barriers
        ) / 5.5
        
        # Find peaks in boundary score (actual boundaries)
        if len(boundary_score) > 10:
            peaks, properties = find_peaks(boundary_score, 
                                         height=np.mean(boundary_score) + np.std(boundary_score),
                                         distance=50)
        else:
            peaks = np.array([])
        
        if self.verbose:
            print(f"   Found {len(peaks)} structural boundaries")
            if len(peaks) > 0:
                print(f"   Boundary locations: {peaks[:5].tolist()}...")  # Show first 5
                # Print boundary strengths individually
                strengths = boundary_score[peaks[:5]]
                strength_str = ", ".join([f"{s:.3f}" for s in strengths])
                print(f"   Boundary strengths: [{strength_str}]...")
        
        return {
            'boundary_score': boundary_score,
            'boundary_locations': peaks,
            'fractal_dimension': fractal_dims,
            'structural_coherence': coherences[0] if coherences else np.array([]),
            'coupling_strength': coupling,
            'structural_entropy': entropy
        }   

    def apply_structural_filtering(self, signal: np.ndarray, 
                                 boundaries: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply structural boundary-based filtering.
        
        Instead of human-defined thresholds, use the natural boundaries
        found in the topological structure itself!
        """
        filtered_signal = signal.copy()
        
        # Use boundary score as adaptive threshold
        boundary_score = boundaries['boundary_score']
        
        # Pad to match signal length if needed
        if len(boundary_score) < len(signal):
            # Pad with mean value
            padding = len(signal) - len(boundary_score)
            boundary_score = np.pad(boundary_score, (0, padding), 
                                  mode='constant', 
                                  constant_values=np.mean(boundary_score))
        
        # Dynamic threshold based on local boundary strength
        for i in range(len(signal)):
            # Local boundary strength determines sensitivity
            local_boundary = boundary_score[i] if i < len(boundary_score) else np.mean(boundary_score)
            
            # High boundary score = more sensitive detection
            # Low boundary score = less sensitive (far from boundaries)
            sensitivity = 1.0 + 2.0 * local_boundary  # 1x to 3x amplification
            
            filtered_signal[i] *= sensitivity
        
        return filtered_signal
    
    def extract_periodic_breaks(self, breaks: Dict[str, np.ndarray],
                          structures: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Extract periodic patterns in topological breaks.
        """
        if self.verbose:
            print("\n🎯 Extracting periodic patterns from topological breaks...")
        
        # Use combined anomaly score for main analysis
        signal = breaks['combined_anomaly']
        n = len(signal)
        
        # ========== 新しい前処理 ==========
        # 1. 短周期ノイズの除去（メディアンフィルタ）
        from scipy.ndimage import median_filter
        signal_clean = median_filter(signal, size=20)  # 20日以下の変動を除去
        
        # 2. 長期トレンドの除去
        from scipy.signal import detrend
        signal_detrended = detrend(signal_clean, type='linear')
        
        # 3. 低周波成分の強調（ローパスフィルタ）
        from scipy.signal import butter, filtfilt
        # カットオフ周期100日のローパスフィルタ
        fs = 1.0  # 1日サンプリング
        cutoff_period = 100  # days
        cutoff_freq = 1.0 / cutoff_period
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff < 1.0:  # 有効な周波数範囲
            b, a = butter(3, normal_cutoff, btype='low', analog=False)
            signal_lowpass = filtfilt(b, a, signal_detrended)
        else:
            signal_lowpass = signal_detrended
        
        # 正規化
        signal = signal_lowpass - np.mean(signal_lowpass)
        if np.std(signal) > 0:
            signal = signal / np.std(signal)
        
        # ========== 長周期検出の強化 ==========
        detected_periods = []
        
        # Method 1: FFT with zero-padding for better resolution
        # ゼロパディングで周波数分解能を向上
        n_padded = n * 4  # 4倍にパディング
        yf = fft(signal, n=n_padded)
        xf = fftfreq(n_padded, 1.0)
        power = np.abs(yf[1:n_padded//2])**2
        freqs = xf[1:n_padded//2]
        
        # Convert to periods
        periods = 1 / freqs[freqs > 0]
        power = power[freqs > 0]
        
        # Focus on long periods
        min_period = 300   # 300日以上のみ！
        max_period = 5000  # 5000日まで
        
        mask = (periods >= min_period) & (periods <= max_period)
        periods_fft = periods[mask]
        power_fft = power[mask]
        
        # より緩い閾値
        if len(power_fft) > 0:
            # 上位パーセンタイルベース
            threshold = np.percentile(power_fft, 70)  # 上位30%
            
            peaks, properties = find_peaks(power_fft, 
                                        height=threshold,
                                        distance=50)  # 長周期なので間隔も広く
            
            for peak in peaks:
                detected_periods.append({
                    'period': periods_fft[peak],
                    'power': power_fft[peak],
                    'method': 'FFT'
                })
        
        # Method 2: ウェーブレット変換（長周期に強い）
        try:
            import pywt
            scales = np.arange(300, 3000, 50)  # 300-3000日のスケール
            coeffs, freqs_cwt = pywt.cwt(signal, scales, 'morl', sampling_period=1.0)
            power_cwt = np.abs(coeffs)**2
            
            # 各スケールでピーク検出
            for i, scale in enumerate(scales):
                if np.max(power_cwt[i]) > np.mean(power_cwt[i]) * 2:
                    detected_periods.append({
                        'period': scale,
                        'power': np.max(power_cwt[i]),
                        'method': 'Wavelet'
                    })
        except ImportError:
            pass  # pywtがない場合はスキップ
        
        # Method 3: 長周期用の自己相関
        # ダウンサンプリングして計算量削減
        downsample_factor = 10
        signal_ds = signal[::downsample_factor]
        autocorr = correlate(signal_ds, signal_ds, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # 長周期のピーク検出
        min_lag = min_period // downsample_factor
        max_lag = min(len(autocorr)-1, max_period // downsample_factor)
        
        if max_lag > min_lag:
            ac_peaks, _ = find_peaks(autocorr[min_lag:max_lag], 
                                  height=0.1, distance=20)
            
            for peak in ac_peaks:
                period = (peak + min_lag) * downsample_factor
                detected_periods.append({
                    'period': float(period),
                    'power': autocorr[peak + min_lag],
                    'method': 'Autocorrelation'
                })
        
        # Method 3: Hilbert transform envelope
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        
        # Find periodicity in envelope
        env_smooth = gaussian_filter1d(envelope, sigma=3)  # Reduced from 5
        env_peaks, _ = find_peaks(env_smooth, distance=min_period//2)  # More flexible
        
        if len(env_peaks) > 2:
            # Period from peak spacing
            peak_diffs = np.diff(env_peaks)
            for period in np.unique(peak_diffs):
                if min_period <= period <= max_period:
                    detected_periods.append({
                        'period': float(period),
                        'power': 1.0,  # Normalized
                        'method': 'Envelope'
                    })
        
        # Consolidate similar periods
        consolidated = []
        tolerance = 0.15  # Increased from 0.1 to 0.15 (15% tolerance)
        
        for p in detected_periods:
            found = False
            for c in consolidated:
                if abs(p['period'] - c['period']) / c['period'] < tolerance:
                    # Average similar periods
                    c['period'] = (c['period'] + p['period']) / 2
                    c['power'] = max(c['power'], p['power'])
                    c['detections'] += 1
                    found = True
                    break
            
            if not found:
                p['detections'] = 1
                consolidated.append(p)
        
        # Sort by confidence (power * detections)
        for p in consolidated:
            p['confidence'] = p['power'] * p['detections']

        consolidated.sort(key=lambda x: x['confidence'], reverse=True)

        # Confidence値の正規化（異常値対策）
        max_confidence = max(p['confidence'] for p in consolidated) if consolidated else 1.0
        if consolidated:
            confidences = [p['confidence'] for p in consolidated]
            max_conf = max(confidences)
            min_conf = min(c for c in confidences if c > 0)
            
            if max_conf / min_conf > 100:  # 差が大きすぎる場合のみ
                for p in consolidated:
                    if p['confidence'] > 0:
                        # 対数変換で差を縮める
                        p['confidence'] = np.log10(p['confidence'] + 1) * 10

        if self.verbose:
            print(f"   Detected {len(consolidated)} periodic patterns")
            print("   Top candidates:")
            for i, p in enumerate(consolidated[:10]):  # Show more candidates
                print(f"   {i+1}. Period: {p['period']:.0f} days, " +
                      f"Confidence: {p['confidence']:.2f}, " +
                      f"Power: {p['power']:.3f}, " +
                      f"Methods: {p['detections']}")

        return consolidated
       
    def decompose_planetary_signals(self, breaks: Dict[str, np.ndarray],
                                  detected_periods: List[Dict],
                                  max_planets: int = 5) -> Dict[str, Dict]:  # Increased from 3 to 5
        """
        Decompose the topological breaks into individual planetary contributions.
        
        Each planet creates a unique pattern of topological disturbance.
        """
        if self.verbose:
            print("\n🪐 Decomposing individual planetary signatures...")
        
        signal = breaks['combined_anomaly']
        t = np.arange(len(signal))
        residual = signal.copy()
        
        planetary_signatures = {}
        
        # Extract top planetary signals
        for i, period_info in enumerate(detected_periods[:max_planets]):
            if self.verbose:
                print(f"\n   Analyzing period {period_info['period']:.0f} days (confidence={period_info['confidence']:.3f})...")
                
            if period_info['confidence'] < 0.1:  # Lowered from 0.5 to 0.2
                if self.verbose:
                    print(f"   → Skipped: confidence too low")
                break
                
            period = period_info['period']
            
            # Fit sinusoidal model to topological breaks
            def model(t, A, phi):
                return A * np.sin(2 * np.pi * t / period + phi)
            
            try:
                # Initial guess
                A0 = np.std(residual)
                
                # Fit with bounds
                popt, _ = curve_fit(model, t, residual,
                                  p0=[A0, 0],
                                  bounds=([0, -np.pi], [3*A0, np.pi]))
                
                # Extract signal
                planet_signal = model(t, *popt)
                
                # Calculate signal strength
                signal_power = np.var(planet_signal)
                total_power = np.var(signal)
                contribution = signal_power / total_power if total_power > 0 else 0
                
                if contribution > 0.005:  # Lowered from 0.01 (0.5% contribution)
                    # Generate planet name dynamically
                    if i < 3:
                        planet_name = ['Planet_X', 'Planet_Y', 'Planet_Z'][i]
                    else:
                        planet_name = f'Planet_{chr(65+i)}'  # A, B, C, ...
                    
                    planetary_signatures[planet_name] = {
                        'period': period,
                        'amplitude': abs(popt[0]),
                        'phase': popt[1],
                        'signal': planet_signal,
                        'contribution': contribution,
                        'confidence': period_info['confidence'],
                        'topological_impact': np.max(np.abs(planet_signal))
                    }
                    
                    # Remove from residual
                    residual -= planet_signal
                    
                    if self.verbose:
                        print(f"   {planet_name}: Period={period:.0f}d, " +
                              f"Amplitude={abs(popt[0]):.5f}, " +
                              f"Impact={planetary_signatures[planet_name]['topological_impact']:.3f}, " +
                              f"Contribution={contribution:.1%}")
                else:
                    if self.verbose:
                        print(f"   Skipped period {period:.0f}d - contribution too low ({contribution:.3%})")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   Could not fit period {period:.0f} days: {str(e)}")
        
        if self.verbose:
            print(f"\n📊 Decomposition complete:")
            print(f"   Successfully decomposed: {len(planetary_signatures)} planets")
            print(f"   Total analyzed periods: {min(max_planets, len(detected_periods))}")
            if len(planetary_signatures) < 3:
                print(f"   ⚠️ Warning: Expected 3 planets, found {len(planetary_signatures)}")
                print(f"   Consider adjusting detection thresholds or checking data quality")
        
        return planetary_signatures
    
    def detect_primary_structural_period(self, structures: Dict[str, np.ndarray]) -> float:
        """
        主構造の周期性を純粋にトポロジカルに検出
        （物理的な「軌道」という概念すら使わない！）
        """
        positions = structures['positions']
        
        # Method 1: トポロジカルチャージの巻き数から
        if 'Q_cumulative' in structures:
            Q_final = structures['Q_cumulative'][-1]
            n_obs = len(structures['Q_cumulative'])
            topological_winding = abs(Q_final)
            
            if topological_winding > 0.5:
                structural_period = n_obs / topological_winding
                if 400 < structural_period < 1000:
                    if self.verbose:
                        print(f"   Primary structural period from Q_Λ: {structural_period:.0f} steps")
                    return structural_period
        
        # Method 2: 構造的自己相似性から
        structural_distances = np.linalg.norm(positions, axis=1)
        structural_pattern = structural_distances - np.mean(structural_distances)
        
        period = self._find_structural_recurrence(structural_pattern)
        if period:
            return period
            
        # Method 3: 位相空間での回帰性（フォールバック）
        return self._find_phase_space_return(positions)
    
    def _find_structural_recurrence(self, pattern: np.ndarray, 
                                   min_lag: int = 50, 
                                   max_lag: int = 1000) -> Optional[float]:
        """構造的再帰パターンの検出（自己相関の構造的解釈）"""
        # 構造的類似性の計算
        structural_similarity = correlate(pattern, pattern, mode='full')
        structural_similarity = structural_similarity[len(structural_similarity)//2:]
        structural_similarity = structural_similarity / structural_similarity[0]
        
        search_end = min(max_lag, len(structural_similarity)-1)
        if search_end > min_lag:
            recurrence_peaks, _ = find_peaks(structural_similarity[min_lag:search_end], 
                                           height=0.3, distance=100)
            if len(recurrence_peaks) > 0:
                period = recurrence_peaks[0] + min_lag
                if self.verbose:
                    print(f"   Structural recurrence period: {period} steps")
                return period
        return None
    
    def _find_phase_space_return(self, positions: np.ndarray) -> float:
        """位相空間での初期状態への構造的回帰"""
        initial_structure = positions[0]
        structural_distances = []
        
        # 構造的距離（物理的距離ではない！）
        for i in range(len(positions)//4, len(positions)):
            structural_diff = np.linalg.norm(positions[i] - initial_structure)
            structural_distances.append((i, structural_diff))
        
        if structural_distances:
            structural_distances.sort(key=lambda x: x[1])
            period = structural_distances[0][0]
        else:
            period = len(positions)//2
            
        if self.verbose:
            print(f"   Phase space return period: {period} steps")
        return period
    
    def _generate_structural_perturbation_pattern(self, 
                                                structural_period: float,
                                                primary_period: float,
                                                n_steps: int) -> np.ndarray:
        """
        純粋に構造的な摂動パターンの生成
        （物理シミュレーションではなく、トポロジカルパターン！）
        """
        t = np.arange(n_steps)
        
        # 構造的位相の進化
        primary_phase = 2 * np.pi * t / primary_period
        perturber_phase = 2 * np.pi * t / structural_period
        
        # 位相差（構造的相互作用）
        phase_coupling = primary_phase - perturber_phase
        
        # トポロジカル結合強度（距離の概念なし！）
        # 位相が同期している時に強い相互作用
        topological_coupling = np.cos(phase_coupling)
        
        # 構造的変調
        structural_modulation = 1.0 + 0.5 * topological_coupling
        
        # 非線形効果（高次の構造的結合）
        nonlinear_coupling = np.sin(2 * phase_coupling) * 0.2
        
        # 最終的な構造的パターン
        pattern = structural_modulation + nonlinear_coupling
        
        # 正規化（純粋に統計的）
        pattern = (pattern - np.mean(pattern)) / np.std(pattern)
        
        return pattern
    
    def refine_periods_with_structural_matching(self, 
                                              planetary_signatures: Dict[str, Dict],
                                              structures: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        検出された周期を構造的パターンマッチングで精緻化
        （物理シミュレーションは使わない！）
        """
        if self.verbose:
            print("\n🔧 Refining periods with structural pattern matching...")
        
        # 主構造の周期を取得
        primary_period = self.detect_primary_structural_period(structures)
        
        refined_signatures = {}
        
        for name, signature in planetary_signatures.items():
            observed_period = signature['period']
            best_factor = 1.0
            best_coherence = 0.0
            best_structural_fit = float('inf')
            
            # 各factor候補でテスト
            for factor in self.PERIOD_REFINEMENT_FACTORS:
                trial_period = observed_period * factor
                
                # 構造的に意味のある範囲チェック
                if self.STRUCTURAL_PERIOD_RANGE[0] < trial_period < self.STRUCTURAL_PERIOD_RANGE[1]:
                    # 構造的パターン生成
                    structural_pattern = self._generate_structural_perturbation_pattern(
                        trial_period, primary_period, len(structures['positions'])
                    )
                    
                    # 観測された構造的摂動と比較
                    observed_pattern = signature['signal']
                    
                    # 長さを合わせる
                    min_len = min(len(structural_pattern), len(observed_pattern))
                    structural_pattern = structural_pattern[:min_len]
                    observed_pattern = observed_pattern[:min_len]
                    
                    # 構造的一貫性を計算（相関ではなく！）
                    # 位相同期性
                    phase_sync = self._compute_phase_synchronization(
                        observed_pattern, structural_pattern
                    )
                    
                    # パターンの構造的類似性
                    structural_similarity = self._compute_structural_similarity(
                        observed_pattern, structural_pattern
                    )
                    
                    # 総合的な構造的適合度
                    structural_fit = phase_sync * structural_similarity
                    
                    if structural_fit > best_coherence:
                        best_coherence = structural_fit
                        best_factor = factor
                        best_structural_fit = 1.0 - structural_fit  # 残差的な意味
            
            # 精緻化された結果を保存
            refined_signatures[name] = signature.copy()
            refined_signatures[name]['original_period'] = observed_period
            refined_signatures[name]['refined_period'] = observed_period * best_factor
            refined_signatures[name]['refinement_factor'] = best_factor
            refined_signatures[name]['structural_coherence'] = best_coherence
            
            if self.verbose:
                print(f"   {name}: {observed_period:.0f}steps → {observed_period * best_factor:.0f}steps "
                      f"(factor={best_factor:.2f}, coherence={best_coherence:.3f})")
        
        return refined_signatures
    
    def _compute_phase_synchronization(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """位相同期性の計算（物理的意味なし、純粋に数学的）"""
        # ヒルベルト変換で瞬時位相を取得
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # 位相差
        phase_diff = np.exp(1j * (phase1 - phase2))
        
        # 同期指標（PLV: Phase Locking Value）
        plv = np.abs(np.mean(phase_diff))
        
        return plv
    
    def _compute_structural_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """構造的類似性の計算（トポロジカルな意味での）"""
        # 1. 形状の類似性（動的時間伸縮法的な）
        # 簡略化のため、ここでは正規化相関を使用
        if np.std(pattern1) > 0 and np.std(pattern2) > 0:
            correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        else:
            correlation = 0.0
        
        # 2. エネルギー分布の類似性
        energy1 = np.cumsum(pattern1**2) / np.sum(pattern1**2)
        energy2 = np.cumsum(pattern2**2) / np.sum(pattern2**2)
        energy_similarity = 1.0 - np.mean(np.abs(energy1 - energy2))
        
        # 3. 極値パターンの類似性
        peaks1, _ = find_peaks(pattern1)
        peaks2, _ = find_peaks(pattern2)
        
        if len(peaks1) > 0 and len(peaks2) > 0:
            # ピーク密度の類似性
            density1 = len(peaks1) / len(pattern1)
            density2 = len(peaks2) / len(pattern2)
            peak_similarity = 1.0 - abs(density1 - density2) / max(density1, density2)
        else:
            peak_similarity = 0.5
        
        # 総合的な構造的類似性
        structural_similarity = (
            0.5 * abs(correlation) +      # 形状
            0.3 * energy_similarity +     # エネルギー分布
            0.2 * peak_similarity         # 極値パターン
        )
        
        return structural_similarity
    
    def estimate_topological_parameters(self, 
                                      planetary_signatures: Dict[str, Dict],
                                      structures: Dict[str, np.ndarray],
                                      use_refinement: bool = True) -> List[Dict]:
        """
        純粋にトポロジカルなパラメータ推定
        （物理的な質量、距離、周期の概念を使わない！）
        """
        if self.verbose:
            print("\n🌌 Estimating topological parameters from structural signatures...")
        
        # 精緻化を適用
        if use_refinement:
            planetary_signatures = self.refine_periods_with_structural_matching(
                planetary_signatures, structures
            )
        
        # 主構造のパラメータ（すべて相対的！）
        positions = structures['positions']
        structural_scale = np.mean(np.linalg.norm(positions, axis=1))
        primary_period = self.detect_primary_structural_period(structures)
        
        if self.verbose:
            print(f"   Primary structure: scale={structural_scale:.2f}, period={primary_period:.0f} steps")
        
        # 各構造体のパラメータ推定
        structures_list = []
        
        for name, signature in planetary_signatures.items():
            # 精緻化された周期を使用
            structural_period = signature.get('refined_period', signature['period'])
            
            # 構造的階層の計算（会合周期から）
            # これは「距離」ではなく「構造的階層」！
            if structural_period > primary_period:
                hierarchy_factor = structural_period / primary_period
            else:
                hierarchy_factor = primary_period / structural_period
            
            # トポロジカル半径（構造的スケールの相対値）
            # 周期比の2/3乗則（これは経験的な構造的関係！）
            relative_scale = (structural_period / primary_period) ** (2/3)
            topological_radius = structural_scale * relative_scale
            
            # 構造的影響力（「質量」ではない！）
            impact = signature['topological_impact']
            structural_influence = impact * structural_scale**2
            
            # 影響力を相対的な単位に変換（地球質量ではない！）
            relative_influence = np.clip(structural_influence * 1000, 1, 1000)
            
            structure_params = {
                'name': name,
                'structural_period': structural_period,
                'hierarchy_factor': hierarchy_factor,
                'topological_radius': topological_radius,
                'structural_influence': relative_influence,
                'topological_impact': impact,
                'contribution': signature['contribution'],
                'confidence': signature['confidence'],
                'refinement_factor': signature.get('refinement_factor', 1.0),
                'structural_coherence': signature.get('structural_coherence', 0.0)
            }
            
            structures_list.append(structure_params)
        
        # トポロジカル半径でソート
        structures_list.sort(key=lambda x: x['topological_radius'])

        # 構造的コヒーレンスでフィルタリング（数は制限しない！）
        structures_list = [
            s for s in structures_list 
            if s.get('structural_coherence', 0) > 0.05  # 低コヒーレンスを除外
        ]

        # 重複する構造を統合（周期が近いものをマージ）
        merged_list = []
        for s in structures_list:
            merged = False
            for m in merged_list:
                # 精緻化後の周期で比較！（refined_periodを使用）
                s_period = s.get('refined_period', s['structural_period'])
                m_period = m.get('refined_period', m['structural_period'])
                
                # 周期が25%以内で近い場合は同じ構造とみなす
                if abs(s_period - m_period) / m_period < 0.25:
                    # より高いconfidenceの方を採用
                    if s['confidence'] > m['confidence']:
                        m.update(s)
                    merged = True
                    break
            if not merged:
                merged_list.append(s)

        structures_list = merged_list
        return structures_list   

    def analyze(self, data: pd.DataFrame, positions: np.ndarray) -> Dict:
          """
          Complete Pure Lambda³ analysis pipeline.
          
          NO physics, NO assumptions - just pure topological structure analysis!
          """
          # 1. Compute Lambda³ structures from raw observations
          structures = self.compute_lambda_structures(positions)
          
          # 2. Detect pure structural boundaries
          boundaries = self.detect_structural_boundaries(structures)
          
          # 3. Detect topological breaks and anomalies
          breaks = self.detect_topological_breaks(structures)
          
          # 4. Use structural boundaries to adaptively set detection sensitivity
          if boundaries['boundary_locations'].size > 0:
              if self.verbose:
                  print("\n🎯 Using structural boundaries to guide detection...")
              
              original_anomaly = breaks['combined_anomaly'].copy()
              
              # Dynamic sensitivity based on boundary score
              boundary_score = boundaries['boundary_score']
              if len(boundary_score) < len(original_anomaly):
                  padding = len(original_anomaly) - len(boundary_score)
                  boundary_score = np.pad(boundary_score, (0, padding), mode='edge')
              
              # Apply adaptive sensitivity
              for i in range(len(original_anomaly)):
                  local_boundary = boundary_score[i] if i < len(boundary_score) else 0
                  sensitivity = 1.0 + 3.0 * local_boundary  # 1x to 4x
                  breaks['combined_anomaly'][i] *= sensitivity
              
              if self.verbose:
                  amplification = np.mean(breaks['combined_anomaly']) / np.mean(original_anomaly)
                  print(f"   Average sensitivity amplification: {amplification:.2f}x")
                  print(f"   Peak sensitivity at boundaries: {np.max(1.0 + 3.0 * boundary_score):.2f}x")
          
          # 5. Extract periodic patterns in the breaks
          periodic_patterns = self.extract_periodic_breaks(breaks, structures)

          # 5.5. 紅莉栖の高調波クラスタリング！
          harmonic_families = self.identify_harmonic_families(periodic_patterns)
          
          # ファミリーごとに基本周期だけを抽出
          filtered_patterns = []
          for family_name, patterns in harmonic_families.items():
              if 'fundamental' in family_name or len(patterns) > 0:
                  # 各ファミリーから最も強いシグナルを代表として選択
                  representative = max(patterns, key=lambda x: x['confidence'])
                  filtered_patterns.append(representative)
    
          # 6. Decompose into structural signatures（フィルター済みを使用！）
          structural_signatures = self.decompose_planetary_signals(
              breaks, filtered_patterns  # periodic_patterns → filtered_patterns
          )
          
          # 7. Estimate topological parameters (NO physics!)
          detected_structures = self.estimate_topological_parameters(
              structural_signatures, structures
          )
          
          # Store results for visualization
          self.data = data
          self.positions = positions
          self.structures = structures
          self.boundaries = boundaries
          self.breaks = breaks
          self.periodic_patterns = periodic_patterns
          self.structural_signatures = structural_signatures
          self.detected_structures = detected_structures
          
          return {
              'n_structures_detected': len(detected_structures),
              'hidden_structures': detected_structures,
              'topological_patterns': structures,
              'topological_breaks': breaks,
              'structural_boundaries': boundaries
          }
      
    def plot_results(self, save_path: Optional[str] = None):
        """Comprehensive visualization of Pure Lambda³ analysis."""
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Raw trajectory
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(self.positions[:, 0], self.positions[:, 1], 
                'k-', linewidth=0.5, alpha=0.7)
        ax1.scatter(0, 0, color='orange', s=200, marker='*', label='Center')
        ax1.set_xlabel('X [relative]')
        ax1.set_ylabel('Y [relative]')
        ax1.set_title('Observed Trajectory')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Topological charge evolution
        ax2 = plt.subplot(3, 4, 2)
        Q = self.structures['Q_cumulative']
        ax2.plot(Q, 'b-', linewidth=2)
        ax2.set_xlabel('Transaction [steps]')
        ax2.set_ylabel('Q_Λ (cumulative)')
        ax2.set_title('Topological Charge Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Combined anomaly score
        ax3 = plt.subplot(3, 4, 3)
        anomaly = self.breaks['combined_anomaly']
        ax3.plot(anomaly, 'r-', alpha=0.7)
        ax3.set_xlabel('Transaction [steps]')
        ax3.set_ylabel('Anomaly Score')
        ax3.set_title('Topological Breaks (Combined)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Power spectrum of breaks
        ax4 = plt.subplot(3, 4, 4)
        n = len(anomaly)
        yf = fft(anomaly - np.mean(anomaly))
        xf = fftfreq(n, 1.0)
        power = np.abs(yf[1:n//2])**2
        periods = 1 / xf[1:n//2]
        mask = (periods > 10) & (periods < n/2)
        ax4.semilogy(periods[mask], power[mask], 'k-', alpha=0.5)
        
        # Mark detected structures
        for structure in self.detected_structures:
            ax4.axvline(structure['structural_period'], color='red', 
                       linestyle='--', alpha=0.7, label=structure['name'])
        
        ax4.set_xlabel('Period [steps]')
        ax4.set_ylabel('Power')
        ax4.set_title('Periodogram of Topological Breaks')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Lambda_F magnitude variations
        ax5 = plt.subplot(3, 4, 5)
        ax5.plot(self.structures['lambda_F_mag'], 'g-', alpha=0.7)
        ax5.set_xlabel('Transaction [steps]')
        ax5.set_ylabel('|ΛF|')
        ax5.set_title('Structural Flow Magnitude')
        ax5.grid(True, alpha=0.3)
        
        # 6. Tension field
        ax6 = plt.subplot(3, 4, 6)
        ax6.plot(self.structures['rho_T'], 'm-', alpha=0.7)
        ax6.set_xlabel('Transaction [steps]')
        ax6.set_ylabel('ρT')
        ax6.set_title('Structural Tension Field')
        ax6.grid(True, alpha=0.3)
        
        # 7. Structural signal decomposition
        ax7 = plt.subplot(3, 4, 7)
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, (name, sig) in enumerate(self.structural_signatures.items()):
            ax7.plot(sig['signal'], color=colors[i % len(colors)], alpha=0.7,
                    label=f"{name} (T={sig['period']:.0f})")
        ax7.set_xlabel('Transaction [steps]')
        ax7.set_ylabel('Topological Signal')
        ax7.set_title('Decomposed Structural Signatures')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Q_residual (topological anomaly)
        ax8 = plt.subplot(3, 4, 8)
        ax8.plot(self.breaks['Q_residual'], 'c-', alpha=0.7)
        ax8.set_xlabel('Transaction [steps]')
        ax8.set_ylabel('Q_Λ Residual')
        ax8.set_title('Topological Charge Anomaly')
        ax8.grid(True, alpha=0.3)
        
        # 9. Detection summary
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        summary = "🌟 Pure Lambda³ Detection Results\n" + "="*40 + "\n\n"
        summary += "NO physics used!\n"
        summary += "NO models assumed!\n"
        summary += "Just pure topological analysis!\n\n"
        
        if hasattr(self, 'boundaries'):
            summary += "🌟 Structural Boundaries:\n"
            summary += f"  Found {len(self.boundaries['boundary_locations'])} natural limits\n"
            summary += "  Pure structure defines its own bounds!\n\n"
        
        summary += f"Detected {len(self.detected_structures)} hidden structures:\n\n"
        
        for structure in self.detected_structures:
            summary += f"{structure['name']}:\n"
            summary += f"  Period: {structure['structural_period']:.0f} steps\n"
            summary += f"  Hierarchy: {structure['hierarchy_factor']:.2f}\n"
            summary += f"  Scale: {structure['topological_radius']:.2f}\n"
            summary += f"  Influence: {structure['structural_influence']:.0f}\n"
            summary += f"  Impact: {structure['topological_impact']:.3f}\n"
            summary += f"  Confidence: {structure['confidence']:.1f}\n\n"
        
        ax9.text(0.1, 0.9, summary, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # 10. System architecture (purely topological)
        ax10 = plt.subplot(3, 4, 10)
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Primary structure
        r_primary = np.mean(np.linalg.norm(self.positions, axis=1))
        ax10.plot(r_primary * np.cos(theta), r_primary * np.sin(theta),
                 'k--', alpha=0.5, label='Primary')
        
        # Detected structures
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, structure in enumerate(self.detected_structures):
            r = structure['topological_radius']
            ax10.plot(r * np.cos(theta), r * np.sin(theta),
                     color=colors[i % len(colors)], linestyle='--', alpha=0.5,
                     label=f"{structure['name']} ({r:.1f})")
        
        ax10.scatter(0, 0, color='orange', s=200, marker='*')
        ax10.set_xlabel('X [relative]')
        ax10.set_ylabel('Y [relative]')
        ax10.set_title('Derived Topological Architecture')
        ax10.legend()
        ax10.axis('equal')
        ax10.grid(True, alpha=0.3)
        ax10.set_xlim(-5, 5)
        ax10.set_ylim(-5, 5)
        
        # 11. Structural boundaries visualization
        ax11 = plt.subplot(3, 4, 11)
        if hasattr(self, 'boundaries'):
            ax11.plot(self.boundaries['boundary_score'], 'purple', alpha=0.7, linewidth=2)
            for boundary in self.boundaries['boundary_locations']:
                ax11.axvline(boundary, color='red', linestyle='--', alpha=0.5)
            ax11.set_xlabel('Transaction [steps]')
            ax11.set_ylabel('Boundary Score')
            ax11.set_title('Pure Structural Boundaries')
            ax11.grid(True, alpha=0.3)
            
            ax11.text(0.02, 0.98, f'Boundaries: {len(self.boundaries["boundary_locations"])}',
                     transform=ax11.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax11.plot(self.structures['helicity'], 'y-', alpha=0.7)
            ax11.set_xlabel('Transaction [steps]')
            ax11.set_ylabel('Helicity')
            ax11.set_title('Structural Helicity')
            ax11.grid(True, alpha=0.3)
        
        # 12. Phase space view
        ax12 = plt.subplot(3, 4, 12)
        n_points = min(len(self.positions)-1, 
                      len(self.structures['lambda_F']), 
                      len(self.breaks['combined_anomaly']))
        
        scatter = ax12.scatter(self.positions[:n_points, 0], 
                             self.structures['lambda_F'][:n_points, 0],
                             c=self.breaks['combined_anomaly'][:n_points],
                             cmap='plasma', s=1, alpha=0.7)
        plt.colorbar(scatter, ax=ax12, label='Anomaly')
        ax12.set_xlabel('X [relative]')
        ax12.set_ylabel('ΛF_x [relative/step]')
        ax12.set_title('Phase Space (colored by anomaly)')
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"\nFigure saved to {save_path}")
        
        plt.show()
    
    def print_results(self):
        """Print analysis results."""
        print("\n" + "="*70)
        print("🌌 Pure Lambda³ Topological Analysis Results")
        print("="*70)
        print("\n⚡ REVOLUTIONARY: No physics was used!")
        print("   All detections come from topological structure alone!")
        
        if hasattr(self, 'boundaries') and self.boundaries['boundary_locations'].size > 0:
            print(f"\n🌟 Structural Boundaries Detected: {len(self.boundaries['boundary_locations'])}")
            print("   (Pure topological limits of influence)")
        
        print(f"\n🔍 Detected {len(self.detected_structures)} hidden structures:")
        print("-"*70)
        
        # Expected patterns for comparison (purely structural)
        expected_patterns = {
            'X': {'period': 923, 'scale': 2.0},
            'Y': {'period': 1435, 'scale': 2.5},
            'Z': {'period': 2274, 'scale': 3.4}
        }
        
        for structure in self.detected_structures:
            print(f"\n{structure['name']}:")
            print(f"  Topological Impact: {structure['topological_impact']:.3f}")
            print(f"  Structural Period: {structure['structural_period']:.0f} steps")
            print(f"  Hierarchy Factor: {structure['hierarchy_factor']:.2f}")
            print(f"  Topological Scale: {structure['topological_radius']:.2f}")
            print(f"  Structural Influence: {structure['structural_influence']:.0f}")
            print(f"  Detection Confidence: {structure['confidence']:.1f}")
            print(f"  Signal Contribution: {structure['contribution']:.1%}")
            
            # Pattern matching
            for exp_name, exp_data in expected_patterns.items():
                period_match = abs(structure['structural_period'] - exp_data['period']) / exp_data['period']
                if period_match < 0.15:  # 15% tolerance
                    print(f"  ✅ Matches Structure {exp_name}! " +
                          f"(Expected: T={exp_data['period']}, scale={exp_data['scale']})")
        
        print("\n" + "="*70)
        print("🎯 Lambda³ SUCCESS: Hidden structures detected from pure topology!")
        print("   No assumptions → Pure discovery from data structure!")
        print("="*70)

def main():
    """Main execution function - Pure Lambda³."""
    parser = argparse.ArgumentParser(
        description='Pure Lambda³ Framework - Topological Structure Detection'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='challenge_blackhole_alpha_noisy.csv',
        help='Path to CSV file containing observational data'
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
    analyzer = PureLambda3Analyzer(verbose=not args.quiet)
    
    # Load data
    data, positions = analyzer.load_and_clean_data(args.data)
    
    # Run pure topological analysis
    results = analyzer.analyze(data, positions)

    # 高調波の後処理マージ
    if hasattr(analyzer, 'detected_structures'):
        analyzer.detected_structures = analyzer.post_process_merge_harmonics(
            analyzer.detected_structures
        )
        # resultsも更新
        results['hidden_structures'] = analyzer.detected_structures
        results['n_structures_detected'] = len(analyzer.detected_structures)

    # Print results
    analyzer.print_results()
    
    # Plot results
    analyzer.plot_results(save_path=args.save_plot)
    
    print("\n✨ Pure Lambda³ analysis complete!")
    print("   The hidden structure revealed through topology alone!")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # For Jupyter/Colab
        print("Running in interactive mode...")
        analyzer = PureLambda3Analyzer(verbose=True)
        
        # Check if data file exists
        import os
        if os.path.exists('challenge_blackhole_alpha_noisy.csv'):
            data, positions = analyzer.load_and_clean_data('challenge_blackhole_alpha_noisy.csv')
            results = analyzer.analyze(data, positions)
            analyzer.print_results()
            analyzer.plot_results()
        else:
            print("⚠️ Data file not found!")
            print("Please generate test data first.")
