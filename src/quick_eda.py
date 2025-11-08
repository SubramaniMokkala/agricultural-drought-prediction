import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('data/drought_dataset_processed.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create comprehensive visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Maharashtra Drought Analysis (2015-2024)', fontsize=18, fontweight='bold', y=0.995)

# 1. NDVI over time with drought periods
ax1 = axes[0, 0]
colors = {'No Drought': 'green', 'Moderate Drought': 'orange', 'Severe Drought': 'red'}
for category in df['drought_category'].unique():
    data = df[df['drought_category'] == category]
    ax1.scatter(range(len(data)), data['ndvi'], label=category, color=colors[category], alpha=0.6, s=50)
ax1.plot(df['ndvi'], color='black', alpha=0.3, linewidth=1)
ax1.axhline(y=0.35, color='red', linestyle='--', alpha=0.5, label='Drought Threshold')
ax1.set_xlabel('Month Index', fontweight='bold')
ax1.set_ylabel('NDVI', fontweight='bold')
ax1.set_title('Vegetation Health (NDVI) Over Time', fontweight='bold', pad=10)
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Precipitation patterns
ax2 = axes[0, 1]
ax2.bar(range(len(df)), df['precipitation_mm'], color=['red' if x == 'Severe Drought' else 'orange' if x == 'Moderate Drought' else 'green' for x in df['drought_category']], alpha=0.7)
ax2.set_xlabel('Month Index', fontweight='bold')
ax2.set_ylabel('Precipitation (mm)', fontweight='bold')
ax2.set_title('Monthly Precipitation with Drought Status', fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

# 3. VCI distribution by drought category
ax3 = axes[1, 0]
df.boxplot(column='vci', by='drought_category', ax=ax3, patch_artist=True)
ax3.set_xlabel('Drought Category', fontweight='bold')
ax3.set_ylabel('Vegetation Condition Index (VCI)', fontweight='bold')
ax3.set_title('VCI Distribution by Drought Category', fontweight='bold', pad=10)
plt.sca(ax3)
plt.xticks(rotation=45)

# 4. Correlation heatmap
ax4 = axes[1, 1]
corr_features = ['ndvi', 'precipitation_mm', 'temp_mean_c', 'vci', 'precip_3month', 'drought_label']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax4, cbar_kws={'label': 'Correlation'})
ax4.set_title('Feature Correlations', fontweight='bold', pad=10)

# 5. Seasonal drought distribution
ax5 = axes[2, 0]
season_counts = pd.crosstab(df['season'], df['drought_category'])
season_counts.plot(kind='bar', stacked=True, ax=ax5, color=['green', 'orange', 'red'])
ax5.set_xlabel('Season', fontweight='bold')
ax5.set_ylabel('Number of Months', fontweight='bold')
ax5.set_title('Drought Distribution by Season', fontweight='bold', pad=10)
ax5.legend(title='Drought Category')
plt.sca(ax5)
plt.xticks(rotation=45)

# 6. NDVI vs Precipitation scatter
ax6 = axes[2, 1]
for category in df['drought_category'].unique():
    data = df[df['drought_category'] == category]
    ax6.scatter(data['precipitation_mm'], data['ndvi'], label=category, color=colors[category], alpha=0.6, s=50)
ax6.set_xlabel('Precipitation (mm)', fontweight='bold')
ax6.set_ylabel('NDVI', fontweight='bold')
ax6.set_title('NDVI vs Precipitation Relationship', fontweight='bold', pad=10)
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/drought_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved: outputs/drought_analysis_comprehensive.png")

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS BY DROUGHT CATEGORY")
print("=" * 60)
print("\nAverage NDVI by drought category:")
print(df.groupby('drought_category')['ndvi'].mean().round(3))

print("\nAverage precipitation by drought category:")
print(df.groupby('drought_category')['precipitation_mm'].mean().round(2))

print("\nAverage VCI by drought category:")
print(df.groupby('drought_category')['vci'].mean().round(2))



plt.show()