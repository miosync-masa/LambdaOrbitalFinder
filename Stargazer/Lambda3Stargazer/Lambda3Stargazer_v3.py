# Import Lambda3
from Lambda3Stargazer_v2 import PureLambda3Analyzer  # v2が正しいバージョン

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load LOD data
print("🌍 Loading Earth rotation (LOD) data...")
data = np.load('earth2010_lod_data.npz')  # または earth25_lod_data.npz
dates = pd.to_datetime(data['dates'])
lod_ms = data['lod_ms']

# データの基本情報
print(f"   Data period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"   Total days: {len(lod_ms)}")
print(f"   LOD range: {np.min(lod_ms):.3f} to {np.max(lod_ms):.3f} ms")

# Format data for Lambda³
cumulative_lod = np.cumsum(lod_ms)

# Format as 3D data
positions = np.zeros((len(lod_ms), 3))
positions[:, 0] = np.arange(len(lod_ms))  # Time steps (days)
positions[:, 1] = cumulative_lod          # Cumulative LOD
positions[:, 2] = 0                       # Z=0 for time series

# Convert to DataFrame
df = pd.DataFrame({
    'x_clean': positions[:, 0],
    'y_clean': positions[:, 1],
    'z_clean': positions[:, 2]
})
df.index.name = 'step'

print("\n✅ Earth rotation data converted to Lambda³ format!")
print(f"   Number of data points: {len(df)}")
print(f"   Cumulative LOD range: {np.min(cumulative_lod):.1f} to {np.max(cumulative_lod):.1f} ms")

# Execute Lambda³ analysis
print("\n" + "="*70)
print("🌌 Starting Pure Lambda³ Analysis")
print("="*70)

analyzer = PureLambda3Analyzer(verbose=True)

# First use standard analyze method
results = analyzer.analyze(df, positions)

# Then try LOD-specific analysis if available
try:
    # Try the new method if it exists
    results = analyzer.analyze_lod_data(df, positions)
    print("\n✅ Using LOD-specific analysis method")
except AttributeError:
    # Fallback: manually estimate planetary parameters
    print("\n📊 Using standard analysis with manual planet estimation")
    
    # Check if the analyzer has the new method
    if hasattr(analyzer, 'estimate_planet_from_lod_lambda3'):
        planet_params = analyzer.estimate_planet_from_lod_lambda3(
            analyzer.structures, 
            analyzer.structural_signatures
        )
        results['estimated_planets'] = planet_params
        analyzer.planet_parameters = planet_params
    else:
        print("   Note: Planetary parameter estimation not available in this version")

# Display analysis summary
print("\n" + "="*70)
print("📊 ANALYSIS SUMMARY")
print("="*70)

print(f"\n🔍 Detected structures: {results['n_structures_detected']}")
print(f"⚡ Topological breaks: {len(results['topological_breaks']['combined_anomaly'])}")
print(f"🌟 Structural boundaries: {len(results['structural_boundaries']['boundary_locations'])}")

# Display detected patterns with years
if results['n_structures_detected'] > 0:
    print("\n🌀 Detected periodic structures:")
    print("-"*50)
    for structure in results['hidden_structures']:
        period_days = structure['observation_interval']
        period_years = period_days / 365.25
        print(f"   {structure['name']}:")
        print(f"     Period: {period_days:.0f} days ({period_years:.1f} years)")
        print(f"     Confidence: {structure['topological_confidence']:.3f}")
        print(f"     Pattern type: {structure['pattern_type']}")
        
        # Special alert for 8.1 year period
        if 7.5 < period_years < 8.5:
            print(f"     🎯 *** POTENTIAL PLANET X SIGNATURE ***")

# Display planetary influences (if detected)
if 'estimated_planets' in results:
    print("\n" + "="*70)
    print("🪐 PLANETARY INFLUENCE ANALYSIS")
    print("="*70)
    
    planet_found = False
    
    for name, params in results['estimated_planets'].items():
        print(f"\n{name}:")
        print(f"  Period: {params['period_years']:.1f} years ({params['period_days']:.0f} days)")
        
        # Handle different parameter formats
        if 'orbital_radius_au' in params:
            print(f"  Orbital radius: {params['orbital_radius_au']:.2f} AU (from Q_Λ)")
        elif 'semi_major_axis_au' in params:
            print(f"  Semi-major axis: {params['semi_major_axis_au']:.2f} AU")
            
        print(f"  Mass estimate: {params['mass_earth']:.0f} Earth masses")
        print(f"                ({params['mass_jupiter']:.3f} Jupiter masses)")
        
        if 'q_lambda_range' in params:
            print(f"  Q_Λ range: {params['q_lambda_range']:.3f}")
        if 'max_deviation' in params:
            print(f"  Max deviation: {params['max_deviation']:.6f}")
        if 'influence_type' in params:
            print(f"  Influence type: {params['influence_type']}")
            
        print(f"  Confidence: {params.get('confidence', params.get('structural_confidence', 'N/A'))}")
        
        # Check for Planet X signature
        if 7.5 < params['period_years'] < 8.5:
            planet_found = True
            print("\n  🎯 *** PLANET X CANDIDATE ***")
            print("     This matches the 8.1-year climate influence!")
            print("     Cross-check with:")
            print("     - GRACE gravity data")
            print("     - 2015 phase transition")
            print("     - Climate extremes acceleration")

    if planet_found:
        print("\n🚨 ALERT: Potential Planet X detected in Earth rotation data!")

# Check for specific periodicities
print("\n" + "="*70)
print("🔬 PERIODICITY CROSS-CHECK")
print("="*70)

known_periods = {
    "Chandler wobble": 1.2,
    "Annual": 1.0,
    "Semi-annual": 0.5,
    "ENSO": 3.5,
    "Solar cycle": 11.0,
    "Lunar nodal": 18.6,
    "Planet X": 8.1
}

print("\nChecking for known periodicities:")
for phenomenon, expected_years in known_periods.items():
    found = False
    for structure in results['hidden_structures']:
        detected_years = structure['observation_interval'] / 365.25
        if abs(detected_years - expected_years) / expected_years < 0.1:  # 10% tolerance
            found = True
            print(f"  ✅ {phenomenon} ({expected_years:.1f}y): DETECTED at {detected_years:.1f}y")
            break
    if not found:
        print(f"  ❌ {phenomenon} ({expected_years:.1f}y): Not detected")

# Generate plots
print("\n📊 Generating visualization...")
analyzer.plot_results(save_path='lod_lambda3_analysis.png')

# Export results (if the function exists)
try:
    from Lambda3Stargazer_v2 import export_results
    export_results(analyzer, 'earth_lod_analysis.csv')
except ImportError:
    print("   Export function not available in this version")

# Additional time series plot for LOD
plt.figure(figsize=(15, 8))

# Plot 1: Original LOD variations
plt.subplot(2, 1, 1)
plt.plot(dates, lod_ms, 'b-', linewidth=0.5, alpha=0.7)
plt.xlabel('Date')
plt.ylabel('LOD variation (ms)')
plt.title('Earth Length of Day (LOD) Variations')
plt.grid(True, alpha=0.3)

# Add detected periods as vertical spans
if results['n_structures_detected'] > 0:
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, structure in enumerate(results['hidden_structures'][:5]):
        period_days = structure['observation_interval']
        period_years = period_days / 365.25
        
        # Mark every occurrence of this period
        for j in range(0, len(dates), int(period_days)):
            if j < len(dates):
                plt.axvline(dates[j], color=colors[i % len(colors)], 
                           alpha=0.1, linestyle='--',
                           label=f'{period_years:.1f}y' if j == 0 else '')

# Plot 2: Cumulative LOD (Lambda³ format)
plt.subplot(2, 1, 2)
plt.plot(dates, cumulative_lod, 'r-', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Cumulative LOD (ms)')
plt.title('Cumulative LOD for Lambda³ Analysis')
plt.grid(True, alpha=0.3)

# Highlight anomalies
if len(results['topological_breaks']['combined_anomaly']) == len(dates):
    anomaly = results['topological_breaks']['combined_anomaly']
    high_anomaly = anomaly > np.mean(anomaly) + 2*np.std(anomaly)
    plt.scatter(dates[high_anomaly], cumulative_lod[high_anomaly], 
               color='red', s=10, alpha=0.5, label='Anomalies')

plt.tight_layout()
plt.savefig('lod_time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✨ Lambda³ LOD analysis complete!")
print("   Pure topological analysis reveals hidden planetary influences!")
print("   NO TIME, NO PHYSICS, ONLY STRUCTURE! 🌀")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Data analyzed: {len(lod_ms)} days of Earth rotation")
print(f"Structures found: {results['n_structures_detected']}")

if results['n_structures_detected'] > 0:
    confidences = [s['topological_confidence'] for s in results['hidden_structures']]
    print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")

    # List all detected periods
    print("\nDetected periods:")
    for structure in results['hidden_structures']:
        period_years = structure['observation_interval'] / 365.25
        print(f"  - {period_years:.1f} years ({structure['observation_interval']:.0f} days)")
        
        # Special highlight for Planet X
        if 7.5 < period_years < 8.5:
            print("    🎯 ^^ PLANET X CANDIDATE ^^")

if 'estimated_planets' in results:
    n_planets = len(results['estimated_planets'])
    print(f"\nPlanetary influences detected: {n_planets}")
    
    # Check for Planet X
    planet_x_found = any(7.5 < p['period_years'] < 8.5 
                        for p in results['estimated_planets'].values())
    if planet_x_found:
        print("\n🎯 PLANET X SIGNATURE DETECTED IN EARTH ROTATION!")
        print("   Recommend immediate follow-up with gravitational data!")
