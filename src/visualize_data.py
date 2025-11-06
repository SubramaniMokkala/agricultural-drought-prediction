import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('data/monthly_precipitation_2023.csv')

# Create visualization
fig, ax = plt.subplots(figsize=(14, 6))

# Bar plot
bars = ax.bar(df['month'], df['total_precipitation_mm'], 
               color=['#d73027' if x < 50 else '#fee08b' if x < 200 else '#1a9850' 
                      for x in df['total_precipitation_mm']],
               edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize plot
ax.set_xlabel('Month', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Precipitation (mm)', fontsize=14, fontweight='bold')
ax.set_title('Maharashtra Monthly Precipitation - 2023\n(Satellite Data: CHIRPS)', 
             fontsize=16, fontweight='bold', pad=20)

# Add drought threshold line
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Drought Risk Threshold')

# Month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(df['month'])
ax.set_xticklabels(month_names, fontsize=11)

# Add legend
ax.legend(fontsize=12)

# Add grid
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig('outputs/precipitation_2023.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: outputs/precipitation_2023.png")

plt.show()

# Print statistics
print("\n" + "=" * 50)
print("PRECIPITATION STATISTICS - 2023")
print("=" * 50)
print(f"Total Annual Rainfall: {df['total_precipitation_mm'].sum():.2f} mm")
print(f"Average Monthly Rainfall: {df['total_precipitation_mm'].mean():.2f} mm")
print(f"Driest Month: {month_names[df['total_precipitation_mm'].idxmin()]} ({df['total_precipitation_mm'].min():.2f} mm)")
print(f"Wettest Month: {month_names[df['total_precipitation_mm'].idxmax()]} ({df['total_precipitation_mm'].max():.2f} mm)")
print(f"Months below drought threshold (<50mm): {len(df[df['total_precipitation_mm'] < 50])}")
print("=" * 50)