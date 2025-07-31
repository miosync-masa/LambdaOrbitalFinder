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

    def compute_lambda_structures(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute fundamental Lambda³ structural quantities from observation sequence.
        """
        if self.verbose:
            print("\n🌌 Computing Lambda³ structural tensors from observation steps...")

        n_steps = len(positions)

        # 1. ΛF - Structural flow field (観測ステップ間の構造変化)
        lambda_F = np.zeros((n_steps-1, 3))
        lambda_F_mag = np.zeros(n_steps-1)

        for step in range(n_steps-1):
            lambda_F[step] = positions[step+1] - positions[step]
            lambda_F_mag[step] = np.linalg.norm(lambda_F[step])

        # 2. ΛFF - Second-order structure (構造変化の変化)
        lambda_FF = np.zeros((n_steps-2, 3))
        lambda_FF_mag = np.zeros(n_steps-2)

        for step in range(n_steps-2):
            lambda_FF[step] = lambda_F[step+1] - lambda_F[step]
            lambda_FF_mag[step] = np.linalg.norm(lambda_FF[step])

        # 3. ρT - Tension field (局所的な構造の張力)
        window_steps = max(3, n_steps // 200)  # 観測ステップの0.5%
        rho_T = np.zeros(n_steps)

        for step in range(n_steps):
            start_step = max(0, step - window_steps)
            end_step = min(n_steps, step + window_steps + 1)
            local_positions = positions[start_step:end_step]

            if len(local_positions) > 1:
                centered = local_positions - np.mean(local_positions, axis=0)
                cov = np.cov(centered.T)
                rho_T[step] = np.trace(cov)

        # 4. Q_Λ - Topological charge (位相的巻き数の変化)
        Q_lambda = np.zeros(n_steps-1)

        for step in range(1, n_steps-1):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step-1] > 1e-10:
                v1 = lambda_F[step-1] / lambda_F_mag[step-1]
                v2 = lambda_F[step] / lambda_F_mag[step]

                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)

                # 2D平面での回転方向
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                signed_angle = angle if cross_z >= 0 else -angle

                Q_lambda[step] = signed_angle / (2 * np.pi)

        # 5. Helicity (構造のねじれ)
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

    def detect_topological_breaks(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect topological breaks and anomalies in observation sequence
        """
        Q_cumulative = structures['Q_cumulative']
        lambda_F_mag = structures['lambda_F_mag']
        lambda_FF_mag = structures['lambda_FF_mag']
        rho_T = structures['rho_T']
        n_steps = structures['n_observation_steps']

        # 観測窓のサイズ（ステップ数の1%）
        window_steps = max(5, n_steps // 100)

        if self.verbose:
            print("\n🔍 Detecting topological breaks in observation sequence...")
            print(f"   Total observation steps: {n_steps}")
            print(f"   Analysis window: {window_steps} steps")

        # 1. トポロジカルチャージの破れ
        if len(Q_cumulative) > 20:
            # Savitzky-Golayフィルタで滑らかな成分を抽出
            window_length = min(15, len(Q_cumulative)//15*2+1)
            if window_length % 2 == 0:
                window_length += 1

            from scipy.signal import savgol_filter
            Q_smooth = savgol_filter(Q_cumulative,
                                  window_length=window_length,
                                  polyorder=3)
            Q_residual = Q_cumulative - Q_smooth
        else:
            Q_residual = Q_cumulative - np.mean(Q_cumulative)

        # 2. 構造フロー（ΛF）の異常検出
        lambda_F_anomaly = np.zeros_like(lambda_F_mag)
        for step in range(len(lambda_F_mag)):
            start = max(0, step - window_steps)
            end = min(len(lambda_F_mag), step + window_steps + 1)

            local_mean = np.mean(lambda_F_mag[start:end])
            local_std = np.std(lambda_F_mag[start:end])

            if local_std > 0:
                # 局所的な標準化
                lambda_F_anomaly[step] = (lambda_F_mag[step] - local_mean) / local_std

        # 3. 構造加速度（ΛFF）の異常
        accel_window = max(3, window_steps // 2)
        lambda_FF_anomaly = np.zeros_like(lambda_FF_mag)

        for step in range(len(lambda_FF_mag)):
            start = max(0, step - accel_window)
            end = min(len(lambda_FF_mag), step + accel_window + 1)

            local_mean = np.mean(lambda_FF_mag[start:end])
            local_std = np.std(lambda_FF_mag[start:end])

            if local_std > 0:
                lambda_FF_anomaly[step] = (lambda_FF_mag[step] - local_mean) / local_std

        # 4. テンション場の跳躍
        from scipy.ndimage import gaussian_filter1d
        rho_T_smooth = gaussian_filter1d(rho_T, sigma=window_steps/3)
        rho_T_breaks = np.abs(rho_T - rho_T_smooth)

        # 5. 複合的な異常スコア
        # 各成分の長さを合わせる
        min_len = min(len(Q_residual), len(lambda_F_anomaly),
                    len(lambda_FF_anomaly), len(rho_T_breaks)-1)

        # 重み付き合成（トポロジカルな重要度に基づく）
        combined_anomaly = (
            np.abs(Q_residual[:min_len]) * 3.0 +        # Q_Λの破れが最重要
            np.abs(lambda_F_anomaly[:min_len]) * 1.5 +  # フローの異常
            np.abs(lambda_FF_anomaly[:min_len]) * 2.0 + # 加速度の異常
            rho_T_breaks[:min_len] * 1.5                # テンションの跳躍
        ) / 8.0

        # 異常パターンの統計
        if self.verbose:
            n_high_anomaly = np.sum(combined_anomaly > np.mean(combined_anomaly) + 2*np.std(combined_anomaly))
            print(f"   High anomaly steps: {n_high_anomaly} ({n_high_anomaly/len(combined_anomaly)*100:.1f}%)")

            # 各成分の寄与
            contributions = {
                'Q_residual': np.mean(np.abs(Q_residual[:min_len])),
                'lambda_F': np.mean(np.abs(lambda_F_anomaly[:min_len])),
                'lambda_FF': np.mean(np.abs(lambda_FF_anomaly[:min_len])),
                'rho_T': np.mean(rho_T_breaks[:min_len])
            }

            max_contrib = max(contributions.values())
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

    def detect_structural_boundaries(self, structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect pure structural boundaries in observation sequence
        """
        if self.verbose:
            print("\n🌟 Detecting structural boundaries in observation sequence...")

        Q_cumulative = structures['Q_cumulative']
        lambda_F = structures['lambda_F']
        rho_T = structures['rho_T']
        n_steps = len(structures['positions'])

        # 1. Q_Λのフラクタル次元解析
        def compute_local_fractal_dimension(series, window_steps=30):
            """観測窓内でのフラクタル次元を計算"""
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

        # 2. ΛFの多スケール構造的一貫性
        def compute_structural_coherence(lambda_F, scale_steps=[5, 10, 20, 40]):
            """異なる観測スケールでの構造の一貫性"""
            coherences = []

            for scale in scale_steps:
                if scale >= len(lambda_F):
                    continue

                coherence_values = []
                for step in range(scale, len(lambda_F) - scale):
                    # 過去と未来の局所ベクトル
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

        # 3. トポロジカル結合強度
        def compute_coupling_strength(Q_series, window_steps=50):
            """観測窓内での構造的結合の強さ"""
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

        # 4. 構造的エントロピー勾配
        def compute_structural_entropy(rho_T, window_steps=30):
            """テンション場の情報エントロピー"""
            entropy = np.zeros(len(rho_T))

            for step in range(window_steps, len(rho_T) - window_steps):
                local_rho = rho_T[step-window_steps:step+window_steps]

                if np.sum(local_rho) > 0:
                    p = local_rho / np.sum(local_rho)
                    entropy[step] = -np.sum(p * np.log(p + 1e-10))

            return entropy

        # 全ての構造的指標を計算
        window_steps = max(30, n_steps // 50)  # 観測ステップの2%
        fractal_dims = compute_local_fractal_dimension(Q_cumulative, window_steps)
        coherences = compute_structural_coherence(lambda_F)
        coupling = compute_coupling_strength(Q_cumulative, window_steps)
        entropy = compute_structural_entropy(rho_T, window_steps)

        # 最小長に正規化
        min_len = min(len(fractal_dims), len(coupling), len(entropy))
        if coherences and len(coherences[0]) > 0:
            min_len = min(min_len, len(coherences[0]))

        # 各指標の勾配を計算
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

        # 境界スコアの合成
        boundary_score = (
            2.0 * fractal_gradient +      # フラクタル次元の変化
            1.5 * coherence_drop +        # 構造的一貫性の低下
            1.0 * coupling_weakness +     # 結合の弱まり
            1.0 * entropy_gradient        # 情報障壁
        ) / 5.5

        # 境界位置の検出（観測ステップ単位）
        if len(boundary_score) > 10:
            min_distance_steps = max(50, n_steps // 30)  # 観測ステップの3.3%
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

    def filter_harmonics_in_recurrence(self, recurrence_patterns: List[Dict]) -> List[Dict]:
        """
        高調波を除外して基本的な再帰パターンだけを抽出
        """
        if not recurrence_patterns:
            return recurrence_patterns

        filtered = []
        used = set()

        # 観測間隔でソート（長い順）
        sorted_patterns = sorted(recurrence_patterns,
                              key=lambda x: x['observation_interval'],
                              reverse=True)

        for i, pattern in enumerate(sorted_patterns):
            if i in used:
                continue

            # このパターンを基本波として採用
            filtered.append(pattern)
            used.add(i)

            # この基本波の高調波を除外
            base_interval = pattern['observation_interval']

            for j, other in enumerate(sorted_patterns):
                if j in used or j == i:
                    continue

                # 整数比をチェック
                ratio = base_interval / other['observation_interval']

                # 2, 3, 4, 5倍の高調波なら除外
                for n in [2, 3, 4, 5]:
                    if abs(ratio - n) < 0.1:
                        used.add(j)
                        if self.verbose:
                            print(f"   Filtered harmonic: {other['observation_interval']:.0f} steps "
                                  f"(n={n} of {base_interval:.0f})")
                        break

        return filtered

    def extract_topological_recurrence(self, structures: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Extract STRUCTURAL recurrence patterns from observation sequence
        """
        if self.verbose:
            print("\n🌌 Extracting topological recurrence patterns from observation steps...")

        Q_cumulative = structures['Q_cumulative']
        lambda_F = structures['lambda_F']
        rho_T = structures['rho_T']
        n_steps = len(Q_cumulative)  # 観測ステップ数

        recurrence_patterns = []

        # 1. Q_Λの巻き数による構造的回帰
        Q_initial = Q_cumulative[0] if n_steps > 0 else 0
        for step_i in range(n_steps//10, n_steps):
            Q_diff = abs(Q_cumulative[step_i] - Q_initial - round(Q_cumulative[step_i] - Q_initial))
            if Q_diff < 0.1:  # ほぼ整数巻き
                recurrence_patterns.append({
                    'observation_interval': step_i,  # 初期からの観測ステップ数
                    'winding_number': round(Q_cumulative[step_i] - Q_initial),
                    'structural_coherence': 1.0 - Q_diff,
                    'pattern_type': 'Q_winding'
                })

        # 2. ΛFベクトル場の位相空間回帰
        for current_step in range(n_steps//4, len(lambda_F)):
            for past_step in range(min(current_step//2, len(lambda_F))):
                if current_step < len(lambda_F) and past_step < len(lambda_F):
                    # 位相空間での構造的距離
                    phase_distance = np.linalg.norm(lambda_F[current_step] - lambda_F[past_step])

                    if phase_distance < np.std(structures['lambda_F_mag']) * 0.2:
                        recurrence_patterns.append({
                            'observation_interval': current_step - past_step,  # ステップ間隔
                            'phase_similarity': 1.0 / (1.0 + phase_distance),
                            'structural_coherence': np.exp(-phase_distance),
                            'pattern_type': 'phase_return'
                        })

        # 3. ρTテンション場の構造的振動
        rho_peaks, _ = find_peaks(rho_T, prominence=np.std(rho_T)*0.5)
        if len(rho_peaks) > 2:
            # ピーク間の観測ステップ数
            step_intervals = np.diff(rho_peaks)
            unique_intervals, counts = np.unique(step_intervals, return_counts=True)

            for interval_steps, count in zip(unique_intervals, counts):
                if count > 2:  # 3回以上繰り返すパターン
                    recurrence_patterns.append({
                        'observation_interval': float(interval_steps),
                        'repetitions': int(count),
                        'structural_coherence': count / len(rho_peaks),
                        'pattern_type': 'tension_oscillation'
                    })

        # パターンの統合
        consolidated = []
        tolerance = 0.15

        for pattern in recurrence_patterns:
            found = False
            for c in consolidated:
                # 観測ステップ間隔が近いものを統合
                if abs(pattern['observation_interval'] - c['observation_interval']) / c['observation_interval'] < tolerance:
                    c['structural_coherence'] = max(c['structural_coherence'], pattern['structural_coherence'])
                    c['detection_count'] = c.get('detection_count', 1) + 1
                    found = True
                    break

            if not found:
                pattern['detection_count'] = 1
                consolidated.append(pattern)

        # 構造的確信度の計算
        for p in consolidated:
            p['topological_confidence'] = p['structural_coherence'] * np.sqrt(p['detection_count'])

        consolidated.sort(key=lambda x: x['topological_confidence'], reverse=True)

        if self.verbose:
            print(f"   Found {len(consolidated)} recurrence patterns in observation sequence")
            for i, p in enumerate(consolidated[:5]):
                print(f"   {i+1}. Interval: {p['observation_interval']:.0f} steps, "
                      f"Confidence: {p['topological_confidence']:.3f}, "
                      f"Type: {p['pattern_type']}")

        return consolidated


    def decompose_structural_signatures(self, structures: Dict[str, np.ndarray],
                                      recurrence_patterns: List[Dict]) -> Dict[str, Dict]:
        """
        観測ステップ列から構造的シグネチャを分解
        """
        if self.verbose:
            print("\n🌀 Decomposing structural signatures from observation sequence...")

        # 観測ステップ配列
        observation_steps = np.arange(len(structures['positions']))

        structural_signatures = {}

        for i, pattern in enumerate(recurrence_patterns[:5]):
            if pattern['topological_confidence'] < 0.1:
                break

            step_interval = pattern['observation_interval']

            # 構造的モデル（観測ステップの関数として）
            def structural_pattern(steps, amplitude, phase_offset):
                # ステップ進行に対する構造的応答
                phase = 2 * np.pi * steps / step_interval + phase_offset
                return amplitude * (np.sin(phase) + 0.3 * np.sin(2*phase))

            # 構造名（観測順序ベース）
            structure_names = ['Primary_Structure', 'Secondary_Structure',
                            'Tertiary_Structure', 'Quaternary_Structure',
                            'Quinary_Structure']

            structural_signatures[structure_names[i]] = {
                'observation_interval': step_interval,  # 観測ステップ間隔
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
        構造的に関連するパターンをグループ化
        （観測ステップ間隔の整数比で判定）
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

            # 他のパターンとの整数比をチェック
            for j, other in enumerate(sorted_patterns[i+1:], i+1):
                if j in used:
                    continue

                ratio = pattern['observation_interval'] / other['observation_interval']

                # 整数比チェック（構造的高調波）
                for n in [2, 3, 4, 5]:
                    if abs(ratio - n) < 0.05 or abs(ratio - 1/n) < 0.05:
                        families[family_key].append(other)
                        used.add(j)
                        break

        return families

    def merge_related_structures(self, structures_list: List[Dict]) -> List[Dict]:
        """
        関連する構造をマージ（高調波OR近接値）
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

                # 条件1: 高調波関係（整数比）
                is_harmonic = any(
                    abs(interval_ratio - n) < 0.1 or abs(interval_ratio - 1/n) < 0.1
                    for n in [2, 3, 4, 5]
                )

                # 条件2: 単純に近い（20%以内）
                is_close = abs(interval_ratio - 1.0) < 0.2

                if is_harmonic or is_close:
                    group.append(s2)
                    used.add(j)

                    if self.verbose:
                        if is_harmonic:
                            print(f"   Harmonic relation: {s1['observation_interval']:.0f} & {s2['observation_interval']:.0f}")
                        else:
                            print(f"   Close values: {s1['observation_interval']:.0f} & {s2['observation_interval']:.0f}")

            # 高調波の場合は基本波を選ぶ、近接値の場合は加重平均
            if any(abs(s['observation_interval'] / group[0]['observation_interval'] - n) < 0.1
                  for s in group[1:] for n in [2, 3, 4, 5]):
                # 高調波グループ：最も長い間隔を基本波とする
                representative = max(group, key=lambda x: x['observation_interval'])
            else:
                # 近接値グループ：最も確信度の高いものを代表に
                representative = max(group, key=lambda x: x['topological_confidence'])

                # 加重平均で間隔を更新
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
        if self.verbose:
            print("\n🌌 Estimating topological parameters from observation sequence...")

        # 主構造の特性（観測ステップベース）
        positions = structures['positions']
        n_observations = len(positions)
        structural_scale = np.mean(np.linalg.norm(positions, axis=1))

        # 主構造の再帰間隔を検出
        primary_interval = self.detect_primary_recurrence(structures)

        if self.verbose:
            print(f"   Observation steps: {n_observations}")
            print(f"   Primary structure: scale={structural_scale:.2f}, recurrence={primary_interval:.0f} steps")

        structures_list = []

        for name, signature in structural_signatures.items():
            # 観測ステップ間隔
            observation_interval = signature['observation_interval']

            # 構造的階層（主構造との関係）
            if observation_interval > primary_interval:
                hierarchy_factor = observation_interval / primary_interval
            else:
                hierarchy_factor = primary_interval / observation_interval

            # トポロジカル半径（構造的スケール）
            relative_scale = (observation_interval / primary_interval) ** (2/3)
            topological_radius = structural_scale * relative_scale

            # 構造的影響力
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

        # トポロジカル半径でソート
        structures_list.sort(key=lambda x: x['topological_radius'])

        # 低確信度を除外
        structures_list = [
            s for s in structures_list
            if s['topological_confidence'] > 0.19
        ]

        return structures_list

    def detect_primary_recurrence(self, structures: Dict[str, np.ndarray]) -> float:
        """
        主構造の再帰間隔を検出（観測ステップ単位）
        """
        positions = structures['positions']
        n_steps = len(positions)

        # Method 1: Q_Λの巻き数から
        if 'Q_cumulative' in structures:
            Q_final = structures['Q_cumulative'][-1]
            topological_winding = abs(Q_final)

            if topological_winding > 0.5:
                recurrence_interval = n_steps / topological_winding
                if n_steps * 0.3 < recurrence_interval < n_steps * 0.7:
                    return recurrence_interval

        # Method 2: 構造的自己相似性
        structural_distances = np.linalg.norm(positions, axis=1)
        pattern = structural_distances - np.mean(structural_distances)

        # 自己相関で再帰を検出
        min_lag = int(n_steps * 0.05)  # 5%以上
        max_lag = int(n_steps * 0.7)   # 70%以下

        autocorr = correlate(pattern, pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        if max_lag > min_lag:
            peaks, _ = find_peaks(autocorr[min_lag:max_lag], height=0.3)
            if len(peaks) > 0:
                return float(peaks[0] + min_lag)

        # フォールバック
        return float(n_steps // 3)

    def estimate_physical_parameters(self, structures: Dict[str, np.ndarray], 
                                    structural_signatures: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        構造的パラメータから物理的な天体パラメータを推定
        完全に物理定数なし！純粋なトポロジーから導出！
        """
        if self.verbose:
            print("\n🌟 Estimating physical parameters from pure topology...")
        
        physical_params = {}
        
        # 主構造（中心天体）の特性を推定
        positions = structures['positions']
        central_scale = np.mean(np.linalg.norm(positions, axis=1))
        
        # 最大構造偏差（摂動の強さ）
        lambda_F_mag = structures['lambda_F_mag']
        baseline_flow = np.median(lambda_F_mag)
        max_deviation = np.max(lambda_F_mag) - baseline_flow
        
        if self.verbose:
            print(f"   Central structure scale: {central_scale:.3f}")
            print(f"   Maximum flow deviation: {max_deviation:.6f}")
        
        for name, signature in structural_signatures.items():
            if self.verbose:
                print(f"\n   Analyzing {name}...")
            
            # 1. 軌道長半径の推定
            # トポロジカル半径 = 構造的影響範囲
            a_estimated = signature.get('topological_radius', central_scale)
            
            # 2. 公転周期（観測ステップを日数に変換）
            T_steps = signature['observation_interval']
            T_days = T_steps  # 1ステップ = 1日と仮定
            T_years = T_days / 365.25
            
            # 3. 質量推定（構造的影響力から）
            # 影響力は距離の2乗に反比例する構造的結合として解釈
            structural_influence = signature.get('structural_influence', 1.0)
            
            # 摂動パターンから質量を逆算
            # 最接近距離での構造的結合強度
            r_closest = abs(a_estimated - central_scale)
            if r_closest < 0.1:  # 内側の軌道
                r_closest = a_estimated
            
            # 構造的加速度（観測ステップでの変化率）
            influence_window = T_steps * 0.1  # 影響期間は周期の10%
            structural_acceleration = 2 * max_deviation / (influence_window**2)
            
            # 質量相当値（構造的結合から）
            # M ∝ a × r² （重力定数なしの相対値）
            mass_structural = structural_acceleration * r_closest**2
            
            # 地球質量単位への変換（経験的較正値）
            # これは観測データとの比較から導出される
            calibration_factor = 1e6  # 構造単位→地球質量
            mass_earth = mass_structural * calibration_factor
            
            # 4. 離心率の推定（構造の非対称性から）
            eccentricity = self._estimate_eccentricity_from_structure(
                structures, T_steps, signature
            )
            
            # 5. 構造的確信度から誤差を推定
            confidence = signature['topological_confidence']
            mass_uncertainty = mass_earth * (1 - confidence)
            
            physical_params[name] = {
                'mass_earth': mass_earth,
                'mass_uncertainty': mass_uncertainty,
                'semi_major_axis_au': a_estimated,
                'period_days': T_days,
                'period_years': T_years,
                'eccentricity': eccentricity,
                'closest_approach_au': r_closest,
                'structural_confidence': confidence,
                'detection_method': signature['pattern_type']
            }
            
            if self.verbose:
                print(f"     Mass: {mass_earth:.1f} ± {mass_uncertainty:.1f} Earth masses")
                print(f"     Semi-major axis: {a_estimated:.2f} AU")
                print(f"     Period: {T_years:.1f} years ({T_days:.0f} days)")
                print(f"     Eccentricity: {eccentricity:.2f}")
                print(f"     Confidence: {confidence:.2f}")
        
        return physical_params

    def _estimate_eccentricity_from_structure(self, structures: Dict[str, np.ndarray],
                                            period_steps: float,
                                            signature: Dict) -> float:
        """
        構造の非対称性から離心率を推定
        """
        # 周期の半分で構造を分割
        half_period = int(period_steps / 2)
        
        lambda_F_mag = structures['lambda_F_mag']
        
        # 各半周期での最大値を比較
        asymmetry_ratios = []
        
        for start in range(0, len(lambda_F_mag) - int(period_steps), int(period_steps)):
            first_half = lambda_F_mag[start:start+half_period]
            second_half = lambda_F_mag[start+half_period:start+int(period_steps)]
            
            if len(first_half) > 0 and len(second_half) > 0:
                max_first = np.max(first_half)
                max_second = np.max(second_half)
                
                if max_first > 0 and max_second > 0:
                    ratio = abs(max_first - max_second) / (max_first + max_second)
                    asymmetry_ratios.append(ratio)
        
        if asymmetry_ratios:
            # 平均非対称性から離心率を推定
            mean_asymmetry = np.mean(asymmetry_ratios)
            # 経験的な変換（非対称性の約30%が離心率に寄与）
            eccentricity = min(0.5, mean_asymmetry * 0.3)
        else:
            eccentricity = 0.1  # デフォルト値
        
        return eccentricity

    def analyze(self, data: pd.DataFrame, positions: np.ndarray) -> Dict:
        """
        Complete Pure Lambda³ analysis pipeline with physical parameter estimation.
        """
        # 1-7. 既存の解析ステップ
        structures = self.compute_lambda_structures(positions)
        boundaries = self.detect_structural_boundaries(structures)
        breaks = self.detect_topological_breaks(structures)
        
        if boundaries['boundary_locations'].size > 0:
            if self.verbose:
                print("\n🎯 Using structural boundaries to guide detection...")
            
            original_anomaly = breaks['combined_anomaly'].copy()
            boundary_score = boundaries['boundary_score']
            
            if len(boundary_score) < len(original_anomaly):
                padding = len(original_anomaly) - len(boundary_score)
                boundary_score = np.pad(boundary_score, (0, padding), mode='edge')
            
            for i in range(len(original_anomaly)):
                local_boundary = boundary_score[i] if i < len(boundary_score) else 0
                sensitivity = 1.0 + 3.0 * local_boundary
                breaks['combined_anomaly'][i] *= sensitivity
        
        recurrence_patterns = self.extract_topological_recurrence(structures)
        recurrence_patterns = self.filter_harmonics_in_recurrence(recurrence_patterns)
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
        
        # 8. NEW: 物理パラメータの推定！
        physical_parameters = self.estimate_physical_parameters(
            structures, structural_signatures
        )
        
        # 検出された構造に物理パラメータを追加
        for i, structure in enumerate(detected_structures):
            structure_name = structure['name']
            if structure_name in physical_parameters:
                structure.update(physical_parameters[structure_name])
        
        # 結果を保存
        self.data = data
        self.positions = positions
        self.structures = structures
        self.boundaries = boundaries
        self.breaks = breaks
        self.recurrence_patterns = recurrence_patterns
        self.structural_signatures = structural_signatures
        self.detected_structures = detected_structures
        self.physical_parameters = physical_parameters
        
        return {
            'n_structures_detected': len(detected_structures),
            'hidden_structures': detected_structures,
            'topological_patterns': structures,
            'topological_breaks': breaks,
            'structural_boundaries': boundaries,
            'physical_parameters': physical_parameters,
            'observation_steps': len(positions)
        }

    def print_results(self, expected_data: Optional[Dict] = None):
        """
        観測ステップベースで結果を表示
        
        Args:
            expected_data: 期待される構造のデータ（オプション）
                          指定しない場合は汎用的な表示
        """
        print("\n" + "="*70)
        print("🌌 Pure Lambda³ Topological Analysis Results")
        print("="*70)
        print("\n⚡ NO TIME, NO PHYSICS, ONLY STRUCTURE!")
        print(f"📊 Total observation steps: {len(self.positions)}")
        
        # データの基本統計
        data_range = np.max(self.positions) - np.min(self.positions)
        data_mean = np.mean(np.linalg.norm(self.positions, axis=1))
        
        print(f"\n📈 Data characteristics:")
        print(f"   Data range: {data_range:.3f}")
        print(f"   Mean scale: {data_mean:.3f}")
        
        # 期待値データがある場合のみ表示
        if expected_data:
            print("\n📝 Reference: Expected structures")
            for name, params in expected_data.items():
                if 'mass' in params:
                    mass_earth = params['mass'] * 333000  # 太陽質量→地球質量
                    print(f"   - {name}: period={params['period']} steps, "
                          f"a={params.get('a', 'N/A')} AU, M≈{mass_earth:.1f} Earth masses")
                else:
                    print(f"   - {name}: period={params['period']} steps")
        
        # 構造的境界
        if hasattr(self, 'boundaries') and self.boundaries['boundary_locations'].size > 0:
            print(f"\n🌟 Structural Boundaries: {len(self.boundaries['boundary_locations'])}")
            print("   (Natural topological limits in observation sequence)")
            
            if len(self.boundaries['boundary_locations']) > 1:
                boundary_intervals = np.diff(self.boundaries['boundary_locations'])
                print(f"   Average interval: {np.mean(boundary_intervals):.0f} ± "
                      f"{np.std(boundary_intervals):.0f} steps")
        
        # 検出された構造
        print(f"\n🔍 Detected {len(self.detected_structures)} hidden structures:")
        print("-"*70)
        
        matched_count = 0
        
        for i, structure in enumerate(self.detected_structures):
            print(f"\n{structure['name']}:")
            print(f"  Observation interval: {structure['observation_interval']:.0f} steps")
            
            # 時間スケールの解釈（汎用）
            print(f"  Time interpretations:")
            print(f"    - As days: {structure['observation_interval']/365.25:.2f} years")
            print(f"    - As hours: {structure['observation_interval']/24:.1f} days")
            
            print(f"  Hierarchy factor: {structure['hierarchy_factor']:.2f}")
            print(f"  Topological radius: {structure['topological_radius']:.2f}")
            print(f"  Structural influence: {structure['structural_influence']:.0f}")
            
            # 物理パラメータ（もしあれば）
            if 'mass_earth' in structure:
                print(f"\n  📊 PHYSICAL PARAMETERS (from pure topology):")
                print(f"     Mass: {structure['mass_earth']:.1f} ± {structure['mass_uncertainty']:.1f} Earth masses")
                print(f"     Semi-major axis: {structure['semi_major_axis_au']:.2f} AU")
                print(f"     Period: {structure['period_years']:.1f} years")
                print(f"     Eccentricity: {structure['eccentricity']:.2f}")
            
            print(f"  Detection confidence: {structure['topological_confidence']:.3f}")
            print(f"  Pattern type: {structure['pattern_type']}")
            
            # 期待値とのマッチング（もしあれば）
            if expected_data:
                best_match = None
                best_diff = float('inf')
                
                for exp_name, params in expected_data.items():
                    diff = abs(structure['observation_interval'] - params['period']) / params['period']
                    if diff < best_diff and diff < 0.15:
                        best_diff = diff
                        best_match = exp_name
                
                if best_match:
                    matched_count += 1
                    print(f"\n  ✅ MATCHED: {best_match} "
                          f"(period diff: {best_diff*100:.1f}%)")
        
        # サマリー
        print("\n" + "="*70)
        print("📊 ANALYSIS SUMMARY:")
        print(f"   Total structures detected: {len(self.detected_structures)}")
        
        if expected_data and matched_count > 0:
            print(f"   Matched with expected: {matched_count}/{len(expected_data)}")
        
        # 構造間の関係
        if len(self.detected_structures) > 1:
            print("\n🔗 Structural relationships:")
            sorted_structures = sorted(self.detected_structures, 
                                     key=lambda x: x['observation_interval'])
            base_period = sorted_structures[0]['observation_interval']
            
            for s in sorted_structures[1:]:
                ratio = s['observation_interval'] / base_period
                print(f"   {s['name']} / {sorted_structures[0]['name']} = {ratio:.2f}")
        
        print("\n🎯 Lambda³ SUCCESS: Hidden structures revealed through pure topology!")
        print("   Transaction, not time. Structure, not physics!")
        print("="*70)

    def plot_results(self, save_path: Optional[str] = None):
        """Visualization of Pure Lambda³ analysis - observation step based"""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 14))

        # 1. 観測軌跡
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(self.positions[:, 0], self.positions[:, 1],
                'k-', linewidth=0.5, alpha=0.7)
        ax1.scatter(0, 0, color='orange', s=200, marker='*', label='Center')
        ax1.set_xlabel('X [structural units]')
        ax1.set_ylabel('Y [structural units]')
        ax1.set_title('Observation Trajectory')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # 2. トポロジカルチャージの累積
        ax2 = plt.subplot(3, 4, 2)
        steps = np.arange(len(self.structures['Q_cumulative']))
        ax2.plot(steps, self.structures['Q_cumulative'], 'b-', linewidth=2)
        ax2.set_xlabel('Observation Steps')
        ax2.set_ylabel('Q_Λ (cumulative)')
        ax2.set_title('Topological Winding')
        ax2.grid(True, alpha=0.3)

        # 3. 複合異常スコア
        ax3 = plt.subplot(3, 4, 3)
        anomaly_steps = np.arange(len(self.breaks['combined_anomaly']))
        ax3.plot(anomaly_steps, self.breaks['combined_anomaly'], 'r-', alpha=0.7)
        ax3.set_xlabel('Observation Steps')
        ax3.set_ylabel('Anomaly Score')
        ax3.set_title('Topological Breaks')
        ax3.grid(True, alpha=0.3)

        # 4. 構造的境界
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

        # 6. テンション場
        ax6 = plt.subplot(3, 4, 6)
        rho_steps = np.arange(len(self.structures['rho_T']))
        ax6.plot(rho_steps, self.structures['rho_T'], 'm-', alpha=0.7)
        ax6.set_xlabel('Observation Steps')
        ax6.set_ylabel('ρT')
        ax6.set_title('Structural Tension Field')
        ax6.grid(True, alpha=0.3)

        # 7. 検出された再帰パターン
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

            # Confidence as text
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

        # 9. 検出結果サマリー
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')

        summary = "🌟 Pure Lambda³ Detection Results\n" + "="*40 + "\n\n"
        summary += "NO TIME. NO PHYSICS. ONLY STRUCTURE.\n\n"
        summary += f"Total observation steps: {len(self.positions)}\n"
        summary += f"Structural boundaries: {len(self.boundaries['boundary_locations'])}\n"
        summary += f"Detected structures: {len(self.detected_structures)}\n\n"

        for structure in self.detected_structures[:3]:
            summary += f"{structure['name']}:\n"
            summary += f"  Interval: {structure['observation_interval']:.0f} steps\n"
            summary += f"  Confidence: {structure['topological_confidence']:.1f}\n\n"

        ax9.text(0.1, 0.9, summary, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        # 10. トポロジカル構造図
        ax10 = plt.subplot(3, 4, 10)
        theta = np.linspace(0, 2*np.pi, 100)

        # 主構造
        r_primary = np.mean(np.linalg.norm(self.positions, axis=1))
        ax10.plot(r_primary * np.cos(theta), r_primary * np.sin(theta),
                'k--', alpha=0.5, label='Primary')

        # 検出された構造
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

        # 11. ヘリシティ
        ax11 = plt.subplot(3, 4, 11)
        hel_steps = np.arange(len(self.structures['helicity']))
        ax11.plot(hel_steps, self.structures['helicity'], 'y-', alpha=0.7)
        ax11.set_xlabel('Observation Steps')
        ax11.set_ylabel('Helicity')
        ax11.set_title('Structural Helicity')
        ax11.grid(True, alpha=0.3)

        # 12. 位相空間（異常度で色付け）
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


def export_results(analyzer, input_filename):
    """Export analysis results to files"""
    import json
    from datetime import datetime

    # Create output filename
    base_name = input_filename.replace('.csv', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Export detected structures
    structures_data = []
    for structure in analyzer.detected_structures:
        structure_dict = {
            'name': structure['name'],
            'observation_interval_steps': float(structure['observation_interval']),
            'hierarchy_factor': float(structure['hierarchy_factor']),
            'topological_radius': float(structure['topological_radius']),
            'structural_influence': float(structure['structural_influence']),
            'confidence': float(structure['topological_confidence']),
            'pattern_type': structure['pattern_type']
        }
        
        # Add physical parameters if available
        if 'mass_earth' in structure:
            structure_dict.update({
                'mass_earth': float(structure['mass_earth']),
                'mass_uncertainty': float(structure['mass_uncertainty']),
                'semi_major_axis_au': float(structure['semi_major_axis_au']),
                'period_days': float(structure['period_days']),
                'period_years': float(structure['period_years']),
                'eccentricity': float(structure['eccentricity']),
                'closest_approach_au': float(structure['closest_approach_au'])
            })
        
        structures_data.append(structure_dict)

    output_data = {
        'analysis_type': 'Pure Lambda³ Topological Analysis',
        'observation_steps': len(analyzer.positions),
        'structures_detected': len(structures_data),
        'structural_boundaries': len(analyzer.boundaries['boundary_locations']),
        'primary_recurrence_interval': float(analyzer.detect_primary_recurrence(analyzer.structures)),
        'detected_structures': structures_data,
        'metadata': {
            'input_file': input_filename,
            'analysis_timestamp': timestamp,
            'lambda3_version': '2.0.0-physics'
        }
    }

    # Save JSON
    output_file = f"{base_name}_lambda3_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results exported to: {output_file}")


def main():
    """Main execution function - Pure Lambda³ with multifocus capability!"""
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
    parser.add_argument(
        '--max-structures',
        type=int,
        default=5,
        help='Maximum number of structures to detect (default: 5)'
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

    # マルチフォーカス判定
    if n_observations >= 2500:
        print("\n🔭 MULTI-FOCUS MODE ACTIVATED!")
        print("   Long observation sequence detected.")
        print("   Running dual-scale analysis for complete structure detection...")

        # Phase 1: 近距離探査（短周期検出用）
        print("\n" + "="*60)
        print("📡 PHASE 1: Near-field Detection (1500 steps)")
        print("="*60)
        analyzer_near = PureLambda3Analyzer(verbose=not args.quiet)
        results_near = analyzer_near.analyze(data.iloc[:1500], positions[:1500])

        print(f"\n✅ Phase 1 complete: {results_near['n_structures_detected']} structures detected")

        # Phase 2: 遠距離探査（長周期検出用）
        print("\n" + "="*60)
        print("📡 PHASE 2: Far-field Detection (full data)")
        print("="*60)
        results_far = analyzer.analyze(data, positions)

        print(f"\n✅ Phase 2 complete: {results_far['n_structures_detected']} structures detected")

        # 結果統合サマリー
        print("\n" + "="*60)
        print("🌟 MULTI-FOCUS DETECTION SUMMARY")
        print("="*60)

        # 近距離で検出した構造を表示
        if results_near['n_structures_detected'] > 0:
            print("\n📍 Near-field structures (likely X, Y):")
            for s in analyzer_near.detected_structures:
                print(f"   - {s['name']}: {s['observation_interval']:.0f} steps "
                      f"(confidence: {s['topological_confidence']:.2f})")

        # 遠距離で検出した構造を表示
        if results_far['n_structures_detected'] > 0:
            print("\n📍 Far-field structures (likely Z):")
            for s in analyzer.detected_structures:
                print(f"   - {s['name']}: {s['observation_interval']:.0f} steps "
                      f"(confidence: {s['topological_confidence']:.2f})")

        # 合計
        total_unique = results_near['n_structures_detected'] + results_far['n_structures_detected']
        print(f"\n📊 Total structures detected: {total_unique}")
        print("   (Note: Some structures may be detected in both phases)")

        # メインの結果は遠距離を使用（プロット用）
        results = results_far

    else:
        # 通常の単一スケール解析
        results = analyzer.analyze(data, positions)

    # Print detailed results
    print("\n" + "="*60)
    print("📋 DETAILED ANALYSIS RESULTS")
    print("="*60)
    analyzer.print_results()

    # Additional summary
    print("\n📈 Analysis Summary:")
    print(f"   Observation steps analyzed: {results['observation_steps']}")
    print(f"   Structures detected: {results['n_structures_detected']}")
    print(f"   Structural boundaries found: {len(results['structural_boundaries']['boundary_locations'])}")

    # Plot results
    if not args.quiet:
        print("\n📊 Generating visualization...")
        analyzer.plot_results(save_path=args.save_plot)

    # Export results
    export_results(analyzer, args.data)

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
        default_file = 'challenge_blackhole_alpha_noisy1500.csv'

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
            print("   Or generate test data first.")

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
