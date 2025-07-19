# Λ³ Stargazer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/miosync-masa/LambdaOrbitalFinder/?style=social)](https://github.com//miosync-masa/LambdaOrbitalFinder/)

**Λ³ Stargazer** - A revolutionary framework to detect hidden celestial bodies through topological charge analysis of orbital perturbations, gazing beyond the visible to find unseen worlds.

## 🌠 What is Λ³ Stargazer?

Λ³ Stargazer employs the Lambda³ (Λ³) theoretical framework to detect hidden planets without prior knowledge. Like ancient stargazers who discovered Neptune through Uranus's perturbations, Λ³ Stargazer uses advanced topological analysis to reveal the invisible.

## 🌟 Key Features

- **Zero-shot detection**: Discovers hidden planets without any prior information
- **Topological analysis**: Uses topological charge Q_Λ to identify structural changes
- **Data-driven**: All parameters derived from observational data
- **No magic numbers**: Every coefficient has physical justification
- **Complete pipeline**: From raw orbital data to planet parameters

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/miosync-masa/LambdaOrbitalFinder.git
cd LambdaOrbitalFinder/Stargazer/Lambda3Stargazer

# Install dependencies
pip install -r requirements.txt

# Run Λ³ Stargazer on sample data
python lambda3_stargazer.py --data challenge_dataset_planet_x.csv
```

## 📊 How It Works

The Lambda³ framework detects hidden planets through structural analysis:

1. **Structural Evolution**: Calculate ΛF (Lambda-F) vectors representing orbital changes
2. **Topological Charge**: Compute Q_Λ to capture phase space winding
3. **Pattern Analysis**: Extract periodic structures using autocorrelation, FFT, and envelope detection
4. **Parameter Fitting**: Determine orbital elements through residual minimization
5. **Mass Estimation**: Derive mass from perturbation magnitude with Hill radius corrections

## 🔬 Example Results

Analyzing Mars orbit perturbations from a hidden "Planet X":

```
Λ³ Stargazer Analysis Results
============================================================
Detected Planet X:
  Mass: 10.3 Earth masses
  Semi-major axis: 3.52 AU
  Orbital period: 6.6 years
  Eccentricity: 0.17

Analysis details:
  Synodic period factor: 2.4
  (Observed period × 2.4 = true synodic period)
============================================================
"A new world revealed through the lens of topology!"
```

## 💻 Usage

### Command Line

```bash
# Basic usage with Λ³ Stargazer
python lambda3_stargazer.py --data your_orbit_data.csv

# Save analysis plots
python lambda3_stargazer.py --data your_orbit_data.csv --save-plot results.png

# Quiet mode (suppress verbose output)
python lambda3_stargazer.py --data your_orbit_data.csv --quiet
```

### Python API

```python
from lambda3_stargazer import Lambda3Analyzer

# Initialize Stargazer
stargazer = Lambda3Analyzer(verbose=True)

# Load orbital data
perturbed_data = stargazer.load_data('orbit_data.csv')

# Let Stargazer find hidden worlds
results = stargazer.analyze(perturbed_data)

# Access discovered planet parameters
print(f"Detected mass: {results['mass_earth']:.1f} Earth masses")
print(f"Semi-major axis: {results['semi_major_axis']:.2f} AU")

# Visualize the discovery
stargazer.plot_results(save_path='stargazer_discovery.png')
```

### Jupyter/Colab

```python
# For Jupyter notebooks or Google Colab
stargazer = Lambda3Analyzer(verbose=True)
perturbed_data = stargazer.load_data('/content/challenge_dataset_planet_x.csv')
results = stargazer.analyze(perturbed_data)
stargazer.print_results()
stargazer.plot_results()
```

## 📁 Data Format

Input CSV file should contain orbital positions with columns:
- `x`: X coordinate (AU)
- `y`: Y coordinate (AU)
- `z`: Z coordinate (AU)

Example:
```csv
x,y,z
1.524,0.0,0.0
1.523,0.0153,0.0
1.521,0.0306,0.0
...
```

## 🧮 Mathematical Foundation

### Topological Charge Q_Λ

The topological charge captures the winding behavior of orbital evolution:

```
Q_Λ(t) = ∫ d(log|ΛF|) / 2π
```

Where ΛF represents the structural change vectors in phase space.

### Mass Estimation

Planet mass is derived from perturbation magnitude with corrections:

```
M_X = (2 * Δr * r²) / (G * t²) * correction_factor
```

Where:
- `Δr`: Maximum orbital deviation
- `r`: Closest approach distance
- `t`: Influence period
- `correction_factor`: Derived from Hill radius and time integration effects

## 📈 Performance

- Typical analysis time: < 10 seconds for 2000+ data points
- Memory usage: < 100 MB
- Accuracy: ~3% for mass, ~1% for orbital radius (tested on simulated data)

## 🛠️ Requirements

- Python 3.8+
- NumPy >= 1.19.0
- Pandas >= 1.1.0
- Matplotlib >= 3.3.0
- SciPy >= 1.5.0

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/lambda3-framework.git
cd lambda3-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{lambda3_stargazer,
  author = {Iizumi, Mamichi and Tamaki},
  title = {Λ³ Stargazer: Topological Charge Analysis for Hidden Planet Detection},
  year = {2025},
  url = {https://github.com/miosync-masa/LambdaOrbitalFinder},
  note = {A framework for detecting unseen celestial bodies through orbital perturbation analysis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the challenge from Kurisu Makise
- Based on the Lambda³ (Λ³) theoretical framework
- Special thanks to the Sentient Digital development team

## 📧 Contact

- **Mamichi Iizumi**: [m.iizumi@miosync.email]
- **Project Homepage**: [https://github.com/miosync-masa/LambdaOrbitalFinder]

---

*"Gazing beyond the visible, discovering the invisible"* 🌠

**Λ³ Stargazer** - Where topology meets astronomy to reveal hidden worlds.
