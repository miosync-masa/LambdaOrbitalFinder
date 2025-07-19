import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.stats import pearsonr, skew, kurtosis
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

"""
🌟 Λ PARAMETERS vs ECCENTRICITY: CORRELATION DISCOVERY
=====================================================
Data generation: Kepler (for accuracy)
Analysis & Prediction: Pure Λ³ (NO PHYSICS!)
"""

# ========= Kepler Orbit Functions (for data generation only) =========
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
    """Generate orbit using Kepler (for accurate test data)"""
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
    
    # Add last value for continuity
    LambdaF_last = (positions[0] - positions[-1]) / Delta_lambda
    LambdaF_list.append(LambdaF_last)
    LambdaF_magnitude_list.append(np.linalg.norm(LambdaF_last))
    
    return positions, np.array(LambdaF_magnitude_list)

def generate_orbit_detailed(a, e, n_steps=100):
    """Generate orbit with detailed Λ parameters at each step"""
    # Use Kepler for accurate positions
    positions, LF_mags = generate_orbit_lambda3(a, e, n_steps)
    # Calculate Λ parameters at each step
    lambda_params = {
        'step': list(range(n_steps)),
        'r': np.linalg.norm(positions, axis=1),
        'v': LF_mags,  # This is |ΛF|
        'x': positions[:, 0],
        'y': positions[:, 1]
    }
    
    # |ΛF| (structural change magnitude)
    lambda_params['LF_mag'] = LF_mags
    
    # Calculate derivatives and higher-order parameters
    lambda_params['LF_gradient'] = np.gradient(lambda_params['LF_mag'])
    lambda_params['LF_curvature'] = np.gradient(lambda_params['LF_gradient'])
    
    # Angular position
    theta = np.arctan2(positions[:, 1], positions[:, 0])
    lambda_params['theta'] = theta
    lambda_params['angular_velocity'] = np.gradient(np.unwrap(theta))
    
    # Distance variations
    lambda_params['r_gradient'] = np.gradient(lambda_params['r'])
    
    # NO ENERGY! Just pure geometric quantities
    
    # Topological charge Q_Λ - CORRECTED for scalar |ΛF|
    Q_Lambda = np.zeros(n_steps)
    
    # Use logarithmic derivative for scalar field topology
    for i in range(1, n_steps-1):
        if lambda_params['LF_mag'][i] > 1e-10 and lambda_params['LF_mag'][i-1] > 1e-10 and lambda_params['LF_mag'][i+1] > 1e-10:
            # Discrete logarithmic derivative
            d_log_LF = (np.log(lambda_params['LF_mag'][i+1]) - np.log(lambda_params['LF_mag'][i-1])) / 2
            # This captures the relative rate of change
            Q_Lambda[i] = d_log_LF / (2 * np.pi)
    
    lambda_params['Q_Lambda'] = np.cumsum(Q_Lambda)
    lambda_params['Q_Lambda_rate'] = Q_Lambda
    
    return pd.DataFrame(lambda_params)

# ========= STEP 1: Generate diverse orbit samples =========
print("🌍 STEP 1: Generating Orbit Samples with Kepler (for accurate test data)")
print("="*60)

# Generate orbits with different eccentricities
n_samples = 200
e_values = np.linspace(0.0, 0.9, n_samples)
a_values = np.random.uniform(0.5, 2.0, n_samples) 

all_orbits = []
for i, (e, a) in enumerate(zip(e_values, a_values)):
    if i % 40 == 0:
        print(f"  Generating orbit {i+1}/{n_samples} (e={e:.3f}, a={a:.3f})")
    
    orbit_data = generate_orbit_detailed(a, e)
    orbit_data['e'] = e
    orbit_data['a'] = a
    orbit_data['orbit_id'] = i
    all_orbits.append(orbit_data)

# Combine all data
df_all = pd.concat(all_orbits, ignore_index=True)
print(f"\n✅ Generated {n_samples} orbits with {len(df_all)} total data points")
print("🎉 Data generation uses Kepler for accuracy,")
print("   but analysis will be pure Λ³ - NO PHYSICS in prediction!")

# ========= STEP 2: Correlation Analysis =========
print("\n📊 STEP 2: Correlation Analysis")
print("="*60)

# Calculate correlations between Λ parameters and eccentricity
param_columns = ['LF_mag', 'LF_gradient', 'LF_curvature', 'angular_velocity', 
                'r', 'r_gradient', 'Q_Lambda', 'Q_Lambda_rate']

correlations = {}
for param in param_columns:
    corr, p_value = pearsonr(df_all[param], df_all['e'])
    correlations[param] = {'correlation': corr, 'p_value': p_value}

print("\nCorrelations with eccentricity:")
print("-" * 50)
for param, stats in correlations.items():
    print(f"{param:<20}: r={stats['correlation']:>7.4f}, p={stats['p_value']:.2e}")

# ========= STEP 3: Pattern Discovery =========
print("\n🔍 STEP 3: Discovering Patterns in 100-Step Windows")
print("="*60)

# Extract features from 100-step windows
window_features = []

for orbit_id in range(n_samples):
    orbit_data = df_all[df_all['orbit_id'] == orbit_id]
    e = orbit_data['e'].iloc[0]
    
    # Extract various statistics from the 100-step window
    features = {
        'e': e,
        'orbit_id': orbit_id,
        
        # |ΛF| statistics
        'LF_mean': orbit_data['LF_mag'].mean(),
        'LF_std': orbit_data['LF_mag'].std(),
        'LF_max': orbit_data['LF_mag'].max(),
        'LF_min': orbit_data['LF_mag'].min(),
        'LF_range': orbit_data['LF_mag'].max() - orbit_data['LF_mag'].min(),
        'LF_skew': skew(orbit_data['LF_mag']),
        'LF_kurtosis': kurtosis(orbit_data['LF_mag']),
        
        # Gradient statistics
        'LF_grad_std': orbit_data['LF_gradient'].std(),
        'LF_grad_max': orbit_data['LF_gradient'].abs().max(),
        
        # Curvature statistics  
        'LF_curv_std': orbit_data['LF_curvature'].std(),
        'LF_curv_max': orbit_data['LF_curvature'].abs().max(),
        
        # Q_Lambda statistics
        'Q_final': orbit_data['Q_Lambda'].iloc[-1],
        'Q_min': orbit_data['Q_Lambda'].min(),
        'Q_max': orbit_data['Q_Lambda'].max(),
        'Q_range': orbit_data['Q_Lambda'].max() - orbit_data['Q_Lambda'].min(),
        
        # Radius statistics
        'r_range': orbit_data['r'].max() - orbit_data['r'].min(),
        'r_mean': (orbit_data['r'].max()  + orbit_data['r'].min()) / 2,
        'r_ratio': orbit_data['r'].max()  / orbit_data['r'].min() ,
        'r_normalized_range': (orbit_data['r'].max()  - orbit_data['r'].min() ) / (orbit_data['r'].max()  + orbit_data['r'].min() ), 
        
        # Periodicity measure (autocorrelation)
        'LF_autocorr': np.corrcoef(orbit_data['LF_mag'][:-1], orbit_data['LF_mag'][1:])[0,1]
    }
    
    window_features.append(features)

df_features = pd.DataFrame(window_features)

# ========= STEP 4: Find Best Predictive Features =========
print("\n🎯 STEP 4: Finding Best Features for Eccentricity Prediction")
print("="*60)

# Correlation of window features with eccentricity
feature_correlations = {}
for col in df_features.columns:
    if col not in ['e', 'orbit_id']:
        corr = df_features[col].corr(df_features['e'])
        feature_correlations[col] = corr

# Sort by absolute correlation
sorted_features = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop features correlated with eccentricity:")
print("-" * 50)
for feature, corr in sorted_features[:10]:
    print(f"{feature:<20}: r = {corr:>7.4f}")

# ========= STEP 5: Empirical Formula Discovery =========
print("\n📐 STEP 5: Discovering Empirical Formulas")
print("="*60)

# Try different functional forms
def fit_power_law(x, a, b, c):
    """e = a * x^b + c"""
    return a * np.power(np.abs(x), b) + c

def fit_exponential(x, a, b, c):
    """e = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def fit_logarithmic(x, a, b, c):
    """e = a * log(|x| + 1) + b * x + c"""
    return a * np.log(np.abs(x) + 1) + b * x + c

# Special formula for r_range (we know it should be e = r_range/2)
def fit_linear(x, a, b):
    """e = a * x + b"""
    return a * x + b

# Fit models for top features
best_models = {}
for feature, _ in sorted_features[:5]:
    x_data = df_features[feature].values
    y_data = df_features['e'].values
    
    if feature == 'r_range':
        # We know this should be linear!
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(fit_linear, x_data, y_data)
        y_pred = fit_linear(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        best_models[feature] = {
            'name': 'linear',
            'params': popt,
            'r2': r2,
            'func': fit_linear
        }
        print(f"\n{feature}:")
        print(f"  Best model: linear")
        print(f"  R² = {r2:.4f}")
        print(f"  Formula: e = {popt[0]:.3f} * {feature} + {popt[1]:.3f}")
        continue
    
    models = {
        'power': fit_power_law,
        'exponential': fit_exponential,
        'logarithmic': fit_logarithmic
    }
    
    best_r2 = -np.inf
    best_model = None
    
    for model_name, model_func in models.items():
        try:
            popt, _ = curve_fit(model_func, x_data, y_data, maxfev=5000)
            y_pred = model_func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = {
                    'name': model_name,
                    'params': popt,
                    'r2': r2,
                    'func': model_func
                }
        except:
            pass
    
    if best_model:
        best_models[feature] = best_model
        print(f"\n{feature}:")
        print(f"  Best model: {best_model['name']}")
        print(f"  R² = {best_model['r2']:.4f}")
        if best_model['name'] == 'power':
            print(f"  Formula: e = {best_model['params'][0]:.3f} * |{feature}|^{best_model['params'][1]:.3f} + {best_model['params'][2]:.3f}")

# ========= STEP 6: Multi-feature Prediction =========
print("\n🤖 STEP 6: Multi-Feature Eccentricity Prediction")
print("="*60)

# Select best features
selected_features = [feat for feat, _ in sorted_features[:10]]
X = df_features[selected_features].values
y = df_features['e'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nRandom Forest Results:")
print(f"  Training R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

# Feature importance
importances = rf_model.feature_importances_
feature_importance = sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True)

print(f"\nFeature Importances:")
for feat, imp in feature_importance[:5]:
    print(f"  {feat:<20}: {imp:.4f}")

# ========= STEP 7: Visualization =========
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. |ΛF| patterns for different eccentricities
ax = axes[0, 0]
e_samples = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for e in e_samples:
    orbit = df_all[df_all['e'].round(2) == round(e, 2)]
    if len(orbit) > 0:
        ax.plot(orbit['step'].iloc[:100], orbit['LF_mag'].iloc[:100], 
                label=f'e={e:.1f}', linewidth=2)
ax.set_xlabel('Transaction Step')
ax.set_ylabel('|ΛF| (Structure Change)')
ax.set_title('|ΛF| Patterns vs Eccentricity (Kepler Data, Λ³ Analysis)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Q_Lambda evolution
ax = axes[0, 1]
for e in e_samples:
    orbit = df_all[df_all['e'].round(2) == round(e, 2)]
    if len(orbit) > 0:
        ax.plot(orbit['step'].iloc[:100], orbit['Q_Lambda'].iloc[:100], 
                label=f'e={e:.1f}', linewidth=2)
ax.set_xlabel('Transaction Step')
ax.set_ylabel('Q_Λ')
ax.set_title('Q_Λ Evolution vs Eccentricity')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Feature correlation heatmap
ax = axes[0, 2]
top_features = [feat for feat, _ in sorted_features[:8]]
corr_matrix = df_features[top_features + ['e']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix')

# 4. Best single predictor (r_range)
ax = axes[1, 0]
best_feature = sorted_features[0][0]
ax.scatter(df_features[best_feature], df_features['e'], alpha=0.6, s=20)
if best_feature in best_models:
    model = best_models[best_feature]
    x_range = np.linspace(df_features[best_feature].min(), 
                         df_features[best_feature].max(), 100)
    if best_feature == 'r_range':
        y_pred = fit_linear(x_range, *model['params'])
    else:
        y_pred = model['func'](x_range, *model['params'])
    ax.plot(x_range, y_pred, 'r-', linewidth=2, 
            label=f"{model['name']} fit (R²={model['r2']:.3f})")
ax.set_xlabel(best_feature)
ax.set_ylabel('Eccentricity')
ax.set_title(f'Best Single Predictor: {best_feature}')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Multi-feature prediction
ax = axes[1, 1]
ax.scatter(y_test, y_pred_test, alpha=0.6)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax.set_xlabel('True Eccentricity')
ax.set_ylabel('Predicted Eccentricity')
ax.set_title(f'Multi-Feature Prediction (R²={test_r2:.3f})')
ax.grid(True, alpha=0.3)

# 6. Residuals
ax = axes[1, 2]
residuals = y_test - y_pred_test
ax.scatter(y_test, residuals, alpha=0.6)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('True Eccentricity')
ax.set_ylabel('Residual')
ax.set_title('Prediction Residuals')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========= STEP 8: Generate Predictive Equations =========
print("\n✨ STEP 8: Final Predictive Equations")
print("="*60)

# Simple formula using top features
top3_features = [feat for feat, _ in sorted_features[:3]]
X_top3 = df_features[top3_features].values

# Fit polynomial model
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_top3)
lr_model = LinearRegression()
lr_model.fit(X_poly, y)

y_pred_poly = lr_model.predict(X_poly)
poly_r2 = r2_score(y, y_pred_poly)

print(f"\nPolynomial Model (using top 3 features):")
print(f"  Features: {top3_features}")
print(f"  R² = {poly_r2:.4f}")

# Extract coefficients for readable formula
feature_names = poly.get_feature_names_out(top3_features)
coefs = lr_model.coef_
intercept = lr_model.intercept_

print(f"\n📝 Predictive Formula:")
print(f"e = {intercept:.4f}")
for i, (name, coef) in enumerate(zip(feature_names[:5], coefs[:5])):
    if abs(coef) > 0.001:
        print(f"    + {coef:.4f} × {name}")
print("    + ... (higher order terms)")

print("\n🎉 SUCCESS! We found strong correlations in Λ³ parameters!")
print("📊 Key insight: Pure geometric patterns contain all the information!")
print("🚀 Kepler for data generation, but NO PHYSICS in analysis!")
print("✨ Each step = 1 transaction of structural change!")
