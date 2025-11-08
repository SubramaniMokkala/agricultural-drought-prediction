import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE ENGINEERING & DROUGHT LABELING")
print("=" * 60)

# Load data
df = pd.read_csv('data/drought_dataset_2015_2024.csv')
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# ============================================
# 1. ROLLING FEATURES (Temporal patterns)
# ============================================
print("\n" + "-" * 60)
print("Creating rolling features...")
print("-" * 60)

# Precipitation rolling sums (cumulative)
df['precip_3month'] = df['precipitation_mm'].rolling(window=3, min_periods=1).sum()
df['precip_6month'] = df['precipitation_mm'].rolling(window=6, min_periods=1).sum()

# NDVI rolling averages
df['ndvi_3month_avg'] = df['ndvi'].rolling(window=3, min_periods=1).mean()

# Precipitation rolling averages
df['precip_3month_avg'] = df['precipitation_mm'].rolling(window=3, min_periods=1).mean()

# Previous month values (lag features)
df['precip_lag1'] = df['precipitation_mm'].shift(1)
df['ndvi_lag1'] = df['ndvi'].shift(1)

print("✓ Created rolling features:")
print("  - 3-month and 6-month precipitation sums")
print("  - 3-month NDVI average")
print("  - Previous month lag features")

# ============================================
# 2. DROUGHT INDICATORS
# ============================================
print("\n" + "-" * 60)
print("Creating drought indicators...")
print("-" * 60)

# Vegetation Condition Index (VCI) - based on NDVI
ndvi_min = df['ndvi'].min()
ndvi_max = df['ndvi'].max()
df['vci'] = ((df['ndvi'] - ndvi_min) / (ndvi_max - ndvi_min)) * 100

# Precipitation anomaly (compared to month's historical average)
monthly_avg_precip = df.groupby('month')['precipitation_mm'].transform('mean')
df['precip_anomaly'] = ((df['precipitation_mm'] - monthly_avg_precip) / monthly_avg_precip) * 100

print("✓ Created drought indicators:")
print("  - VCI (Vegetation Condition Index)")
print("  - Precipitation anomaly")

# ============================================
# 3. DROUGHT LABELS (Target variable)
# ============================================
print("\n" + "-" * 60)
print("Defining drought conditions...")
print("-" * 60)

# Multi-criteria drought definition:
# DROUGHT if ANY of these conditions:
# 1. VCI < 35 (severe vegetation stress)
# 2. Precipitation < 20mm AND precip_3month < 60mm
# 3. NDVI < 0.30 (very low vegetation)

def label_drought(row):
    """Label drought based on multiple indicators"""
    
    # Severe drought
    if (row['vci'] < 25 or 
        (row['precipitation_mm'] < 15 and row['precip_3month'] < 50) or
        row['ndvi'] < 0.27):
        return 2  # Severe drought
    
    # Moderate drought
    elif (row['vci'] < 40 or 
          (row['precipitation_mm'] < 30 and row['precip_3month'] < 100) or
          row['ndvi'] < 0.35):
        return 1  # Moderate drought
    
    # No drought
    else:
        return 0  # No drought

df['drought_label'] = df.apply(label_drought, axis=1)

# Drought category names
drought_categories = {0: 'No Drought', 1: 'Moderate Drought', 2: 'Severe Drought'}
df['drought_category'] = df['drought_label'].map(drought_categories)

print("✓ Drought labels created")
print("\nDrought distribution:")
print(df['drought_category'].value_counts().sort_index())
print(f"\nPercentages:")
for label, count in df['drought_category'].value_counts().sort_index().items():
    pct = (count / len(df)) * 100
    print(f"  {label}: {count} months ({pct:.1f}%)")

# ============================================
# 4. SEASONAL FEATURES
# ============================================
print("\n" + "-" * 60)
print("Adding seasonal features...")
print("-" * 60)

# Season categorization
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

df['season'] = df['month'].apply(get_season)

print("✓ Added season feature")
print("\nDrought by season:")
season_drought = pd.crosstab(df['season'], df['drought_category'], normalize='index') * 100
print(season_drought.round(1))

# ============================================
# 5. SAVE PROCESSED DATASET
# ============================================

# Fill missing temperature values with median (for months without data)
df['temp_mean_c'].fillna(df['temp_mean_c'].median(), inplace=True)
df['temp_max_c'].fillna(df['temp_max_c'].median(), inplace=True)
df['temp_min_c'].fillna(df['temp_min_c'].median(), inplace=True)

# Save processed dataset
df.to_csv('data/drought_dataset_processed.csv', index=False)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 60)
print(f"Final dataset shape: {df.shape}")
print(f"Total features: {len(df.columns)}")
print(f"Saved to: data/drought_dataset_processed.csv")

print("\nFinal dataset columns:")
print(df.columns.tolist())

print("\nSample of processed data:")
print(df[['date', 'ndvi', 'precipitation_mm', 'vci', 'precip_3month', 'drought_category']].head(15))

print("\n" + "=" * 60)