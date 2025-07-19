import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

"""
🌌 100 Orbital Samples Analysis
================================
Generate 100 different orbits and predict Q_Λ from 100-step observations!
"""

# ========= Kepler Orbit Functions (from previous code) =========
def kepler_equation(E, M, e):
    return E - e * np.sin(E) - M

def kepler_equation_derivative(E, M, e):
    return 1 - e * np.cos(E)

def solve_kepler(M, e, tol=1e-12):
    if M < np.pi:
        E0 = M + e/2
    else:
        E0 = M - e/2
    E = newton(kepler_equation, E0, fprime=kepler_equation_derivative, 
               args=(M, e), tol=tol)
    return E

def generate_orbit_lambda3(a, e, n_steps):
    """Generate orbit and calculate Λ³ parameters"""
    positions = []
    
    # Generate positions
    for step in range(n_steps):
        theta = 2 * np.pi * step / n_steps
        
        # Mean anomaly
        M = theta
        
        # Solve for eccentric anomaly
        E = solve_kepler(M, e)
        
        # True anomaly
        true_anomaly = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        
        # Distance from focus
        r = a * (1 - e * np.cos(E))
        
        # Cartesian coordinates
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        positions.append([x, y, 0])
    
    positions = np.array(positions)
    
    # Calculate Λ³ parameters
    Delta_lambda = 1.0
    LambdaF_list = []
    LambdaF_magnitude_list = []
    
    for n in range(1, n_steps):
        LambdaF = (positions[n] - positions[n-1]) / Delta_lambda
        LambdaF_list.append(LambdaF)
        LambdaF_magnitude_list.append(np.linalg.norm(LambdaF))
    
    # Calculate Q_Λ
    Q_Lambda = [0]
    for i in range(1, len(LambdaF_magnitude_list)-1):
        phase_diff = np.arctan2(LambdaF_magnitude_list[i+1], LambdaF_magnitude_list[i]) - \
                    np.arctan2(LambdaF_magnitude_list[i], LambdaF_magnitude_list[i-1])
        if phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        elif phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        Q_Lambda.append(phase_diff / (2 * np.pi))
    Q_Lambda = np.cumsum(Q_Lambda)
    
    return positions, LambdaF_magnitude_list, Q_Lambda

# ========= Generate 100 Orbital Samples =========
print("🚀 Generating 100 orbital samples...")
print("="*60)

np.random.seed(42)  # For reproducibility
n_samples = 100
n_observe = 100  # First 100 steps for observation

# Generate random orbital parameters
# Semi-major axis: 0.3 to 5 AU (Mercury to Jupiter range)
a_values = np.random.uniform(0.3, 5.0, n_samples)

# Eccentricity: 0 to 0.9 (bound orbits)
# Use beta distribution to have more circular orbits
e_values = np.random.beta(2, 5, n_samples) * 0.9

# Period approximation (Kepler's 3rd law)
periods = np.sqrt(a_values**3) * 365.25

results = []
all_steps = []
print(f"Generating orbits with:")
print(f"  a ∈ [{a_values.min():.2f}, {a_values.max():.2f}] AU")
print(f"  e ∈ [{e_values.min():.3f}, {e_values.max():.3f}]")

for i in range(n_samples):
    if i % 20 == 0:
        print(f"  Processing orbit {i+1}/{n_samples}...")
    
    # Use adaptive number of steps based on period
    n_steps = max(int(periods[i]), 365)
    
    # Generate orbit
    positions, LF_mags, Q_Lambda = generate_orbit_lambda3(
        a_values[i], e_values[i], n_steps
    )
    
    # 全step保存：for iループの直前で all_steps = [] しておく
    LambdaF_vectors = [positions[n] - positions[n-1] for n in range(1, len(positions))]
    minlen = min(len(LF_mags), len(Q_Lambda))
    for step in range(1, minlen+1):
        all_steps.append({
            'orbit_id': i,
            'step': step,
            'x': positions[step,0],
            'y': positions[step,1],
            'z': positions[step,2],
            'LambdaF_x': LambdaF_vectors[step-1][0],
            'LambdaF_y': LambdaF_vectors[step-1][1],
            'LambdaF_z': LambdaF_vectors[step-1][2],
            'LambdaF_mag': LF_mags[step-1],
            'Q_Lambda': Q_Lambda[step-1]
        })

    # Extract features from first 100 steps
    if len(LF_mags) >= n_observe:
        LF_obs = np.array(LF_mags[:n_observe])
        Q_obs = Q_Lambda[:n_observe]
        
        # Calculate features
        result = {
            'orbit_id': i,
            'a': a_values[i],
            'e': e_values[i],
            'period': periods[i],
            # |ΛF| features
            'LF_mean': np.mean(LF_obs),
            'LF_std': np.std(LF_obs),
            'LF_range': np.max(LF_obs) - np.min(LF_obs),
            'LF_relative_range': (np.max(LF_obs) - np.min(LF_obs)) / np.mean(LF_obs),
            # Q_Λ features
            'Q_min': np.min(Q_obs),
            'Q_min_position': np.argmin(Q_obs),
            'Q_at_100': Q_obs[-1] if len(Q_obs) >= n_observe else Q_obs[-1],
            'Q_final': Q_Lambda[-1],
            # Additional features
            'Q_range_100': np.max(Q_obs) - np.min(Q_obs),
            'Q_slope_end': (Q_obs[-1] - Q_obs[-10]) / 10 if len(Q_obs) >= 10 else 0
        }
        results.append(result)

df_results = pd.DataFrame(results)
print(f"\n✅ Generated {len(df_results)} valid orbital samples!")

df_steps = pd.DataFrame(all_steps)
df_steps.to_csv("/content/lambda3_orbits_allsteps.csv", index=False)
print("✅ 全軌道・全step分CSV保存完了！")

# ========= Feature Engineering =========
print("\n🔧 Feature Engineering...")

# Create feature matrix
feature_columns = [
    'e', 'a', 'LF_mean', 'LF_std', 'LF_relative_range',
    'Q_min', 'Q_at_100', 'Q_min_position', 'Q_range_100', 'Q_slope_end'
]

X = df_results[feature_columns].values
y = df_results['Q_final'].values

# Add interaction features
X_extended = np.column_stack([
    X,
    X[:, 0] * np.abs(X[:, 5]),  # e × |Q_min|
    X[:, 4] * X[:, 5],  # LF_range × Q_min
    X[:, 0]**2,  # e²
    np.abs(X[:, 5])**0.5  # sqrt(|Q_min|)
])

feature_names_extended = feature_columns + [
    'e×|Q_min|', 'LF_range×Q_min', 'e²', 'sqrt|Q_min|'
]

# ========= Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X_extended, y, test_size=0.2, random_state=42
)

print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

# ========= Model Training =========
print("\n🤖 Training Prediction Models...")
print("="*60)

models = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear'] = lr

# 2. Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
models['Polynomial'] = (lr_poly, poly)

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
models['RandomForest'] = rf

# ========= Model Evaluation =========
print("\n📊 Model Performance on Test Set:")
print("-"*60)
print(f"{'Model':<15} {'R²':<10} {'MAE':<15} {'Rel. Error':<10}")
print("-"*60)

best_model = None
best_r2 = -np.inf

for name, model in models.items():
    if name == 'Polynomial':
        model_obj, poly_transform = model
        y_pred = model_obj.predict(X_test_poly)
    else:
        y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rel_error = np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + 1e-10)) * 100
    
    print(f"{name:<15} {r2:<10.3f} {mae:<15.6f} {rel_error:<10.1f}%")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = (name, model)

# ========= Feature Importance (Random Forest) =========
print("\n🎯 Feature Importance (Random Forest):")
print("-"*50)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

all_steps = []  
for i in range(min(10, len(feature_names_extended))):
    idx = indices[i]
    print(f"{feature_names_extended[idx]:<20}: {importances[idx]:.3f}")

# ========= Visualization =========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Q_final vs eccentricity
ax = axes[0, 0]
scatter = ax.scatter(df_results['e'], df_results['Q_final'], 
                    c=df_results['a'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Eccentricity')
ax.set_ylabel('Q_final')
ax.set_title('Q_final vs Eccentricity (color = semi-major axis)')
plt.colorbar(scatter, ax=ax, label='a [AU]')

# 2. Q_final vs |Q_min|
ax = axes[0, 1]
ax.scatter(np.abs(df_results['Q_min']), df_results['Q_final'], alpha=0.6)
ax.set_xlabel('|Q_min|')
ax.set_ylabel('Q_final')
ax.set_title('Q_final vs |Q_min|')
ax.set_xscale('log')
ax.set_yscale('log')

# 3. Prediction vs Actual
ax = axes[0, 2]
if best_model[0] == 'Polynomial':
    y_pred_best = best_model[1][0].predict(X_test_poly)
else:
    y_pred_best = best_model[1].predict(X_test)

ax.scatter(y_test, y_pred_best, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Actual Q_final')
ax.set_ylabel('Predicted Q_final')
ax.set_title(f'Best Model: {best_model[0]} (R² = {best_r2:.3f})')

# 4. Residuals
ax = axes[1, 0]
residuals = y_test - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.6)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Q_final')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')

# 5. Q_min vs e×|Q_min|
ax = axes[1, 1]
ax.scatter(df_results['e'] * np.abs(df_results['Q_min']), 
          df_results['Q_final'], alpha=0.6)
ax.set_xlabel('e × |Q_min|')
ax.set_ylabel('Q_final')
ax.set_title('Q_final vs e×|Q_min| interaction')

# 6. Distribution of Q_final
ax = axes[1, 2]
ax.hist(df_results['Q_final'], bins=30, alpha=0.7, edgecolor='black')
ax.set_xlabel('Q_final')
ax.set_ylabel('Count')
ax.set_title('Distribution of Q_final')
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ========= Empirical Formula =========
print("\n📐 Empirical Relationships Found:")
print("="*60)

# Fit simple power law: Q_final ~ A × |Q_min|^α × f(e)
from scipy.optimize import curve_fit

def power_law_model(X, A, alpha, beta):
    """Q_final = A × |Q_min|^alpha × exp(beta × e)"""
    Q_min_abs = np.abs(X[:, 0])
    e = X[:, 1]
    return A * (Q_min_abs ** alpha) * np.exp(beta * e)

X_simple = np.column_stack([df_results['Q_min'].values, df_results['e'].values])
popt, _ = curve_fit(power_law_model, X_simple, df_results['Q_final'].values, 
                   p0=[1.0, 0.5, 0.1])

print(f"\nPower Law Model:")
print(f"Q_final ≈ {popt[0]:.3f} × |Q_min|^{popt[1]:.3f} × exp({popt[2]:.3f} × e)")

# Test on specific cases
test_cases = [
    {'name': 'Earth-like', 'e': 0.0167, 'Q_min': -0.00002253},
    {'name': 'Mars-like', 'e': 0.0934, 'Q_min': -0.00005952},
    {'name': 'Mercury-like', 'e': 0.2056, 'Q_min': -0.00106811},
    {'name': 'Comet-like', 'e': 0.7, 'Q_min': -0.00035838}
]

print(f"\nTest on known orbits:")
print("-"*60)
for case in test_cases:
    X_case = np.array([[case['Q_min'], case['e']]])
    Q_pred = power_law_model(X_case, *popt)[0]
    print(f"{case['name']:<15}: Q_final_pred = {Q_pred:.8f}")

print("\n✨ With 100 samples, we can predict Q_final from 100-step observations!")
