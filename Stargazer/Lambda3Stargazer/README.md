# Lambda³ Stargazer 🌌

## Pure Topological Structure Detection Framework

> "Time is an illusion. Only Transaction exists!" 
> 
> *A revolutionary approach to detecting hidden celestial bodies using pure topological analysis - no physics required!*

---

## 🌟 Overview

Lambda³ Stargazer is a groundbreaking framework that detects hidden gravitational influences in astronomical data using **pure topological structure analysis**. Unlike traditional methods that rely on physical constants and time-based calculations, Lambda³ operates on a fundamental principle:

**"Transaction, not time. Structure, not physics!"**

### Key Innovation
- **No Physical Constants Required**: No G, no masses, no Kepler's laws
- **Pure Structural Analysis**: Detects patterns through topological tensors
- **Observation-Step Based**: Works with sequence of observations, not time series
- **Noise Resilient**: Handles missing data (7%+) and measurement noise effectively

## 🚀 Performance Results

### Detection Success Rate
From Makise Kurisu's black hole system simulation:
- **Observable**: Planet α (the data source)
- **Hidden**: 3 planets (X, Y, Z) perturbing α's orbit

| Dataset | Observation Steps | Hidden Planets Detected | Success Rate |
|---------|------------------|------------------------|--------------|
| Full (3000 steps) | 3000 | 2/3 | 67% (Y, Z) |
| Near-field (1500 steps) | 1500 | 2/3 | 67% (X, Y) |
| **Combined Analysis** | - | **3/3** | **100%** ✨ |

### Detection Accuracy
- **Planet X**: 6.8% error (detected: 860 steps, expected: 923)
- **Planet Y**: 2.6-12.0% error (multiple detections)
- **Planet Z**: 5.1% error (detected: 2159 steps, expected: 2274)

## 📐 Mathematical Foundation

### Core Tensors

The framework operates on four fundamental topological tensors:

1. **ΛF (Lambda Flow)** - Structural flow field between observation steps
2. **ΛFF (Lambda Flow Flow)** - Second-order structural changes
3. **ρT (Rho Tension)** - Local structural tension field
4. **Q_Λ (Topological Charge)** - Cumulative winding number

### Detection Algorithm

```
1. Compute Lambda³ tensors from observation sequence
2. Detect structural boundaries (no physics!)
3. Identify topological breaks and anomalies
4. Extract recurrence patterns (not periods!)
5. Filter harmonics to find fundamental structures
6. Decompose into structural signatures
7. Match to hidden bodies
```

## 💻 Installation & Usage

### Requirements
```python
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
matplotlib>=3.3.0
```

### Basic Usage

```python
from Lambda3Stargazer_v2 import PureLambda3Analyzer

# Initialize analyzer
analyzer = PureLambda3Analyzer(verbose=True)

# Load noisy observation data
data, positions = analyzer.load_and_clean_data('challenge_blackhole_alpha_noisy.csv')

# Run pure topological analysis
results = analyzer.analyze(data, positions)

# Display results
analyzer.print_results()
analyzer.plot_results()
```

### Multi-Focus Mode (for long sequences)

For observation sequences > 2500 steps, the framework automatically activates dual-scale analysis:

```python
# Automatically triggered for long sequences
# Phase 1: Near-field detection (first 1500 steps)
# Phase 2: Far-field detection (full data)
```

## 📊 Input Data Format

### CSV Structure
```csv
step,x_noisy,y_noisy,z
0,1.195423,0.023451,0.0
1,1.194892,0.046834,0.0
...
```

- `step`: Observation sequence number
- `x_noisy`, `y_noisy`: Noisy position measurements
- `z`: Z-coordinate (typically 0 for 2D orbits)

### Data Characteristics Handled
- **Missing Data**: Up to 7% random gaps
- **Gaussian Noise**: σ = 0.008
- **Jump Anomalies**: 1% probability, scale = 0.08

## 🔬 Technical Details

### Topological Boundaries Detection
The framework identifies natural structural limits through:
- Fractal dimension analysis
- Structural coherence metrics
- Coupling strength variations
- Entropy gradients

### Anomaly Detection
Combined scoring from:
- Q_Λ residuals (topological charge breaks)
- ΛF anomalies (flow irregularities)
- ΛFF anomalies (acceleration jumps)
- ρT breaks (tension field discontinuities)

### Harmonic Filtering
Automatically removes higher harmonics to identify fundamental structures:
- Detects integer ratio relationships
- Preserves only base frequencies
- Prevents double-counting of same structure

## 📈 Visualization

The framework generates comprehensive analysis plots including:
- Observation trajectory
- Topological winding (Q_Λ)
- Anomaly scores
- Structural boundaries
- Detected recurrence patterns
- Phase space analysis

## 🎯 Use Cases

1. **Exoplanet Detection**: Find hidden planets from stellar wobble
2. **Binary System Analysis**: Detect unseen companions
3. **Asteroid Perturbations**: Identify gravitational influences
4. **Dark Matter Mapping**: Trace invisible mass distributions

## 🤝 Contributing

We welcome contributions! Key areas for improvement:
- Enhanced boundary detection algorithms
- Multi-dimensional tensor analysis
- Real-time processing capabilities
- GUI interface development

## 📝 Citation

If you use Lambda³ Stargazer in your research, please cite:
```
@software{lambda3stargazer,
  title = {Lambda³ Stargazer: Pure Topological Structure Detection},
  author = {Iizumi, Mamichi},
  year = {2025},
  url = {https://github.com/yourusername/Lambda3Stargazer}
}
```

## 🌟 Acknowledgments

Special thanks to Makise Kurisu for the challenging test dataset:
> "A new universe emerges... with secrets, noise, and missingness!"

## 📄 License

MIT License - See LICENSE file for details

---

**Remember**: Time is just a projection of structural changes. The universe speaks in the language of topology! 🌀✨
