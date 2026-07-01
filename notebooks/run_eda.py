import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv('dataset/hdi_dataset.csv')

# Create static/images directory
os.makedirs('static/images', exist_ok=True)

# 1. Schooling vs HDI Strip Plot
plt.figure(figsize=(10, 6))
# Bin the mean years of schooling to make it clean for strip plot
bins = pd.cut(df['Mean years of schooling'], bins=5)
sns.stripplot(x=bins, y='HDI Score', data=df, hue='HDI Tier', palette='viridis', jitter=0.25)
plt.title('Mean Years of Schooling vs HDI Score')
plt.xlabel('Mean Years of Schooling (Binned)')
plt.ylabel('HDI Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/images/schooling_vs_hdi.png', bbox_inches='tight', dpi=100)
plt.close()
print("Saved schooling_vs_hdi.png")

# 2. Life Expectancy vs HDI Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Life expectancy', y='HDI Score', data=df, hue='HDI Tier', palette='coolwarm')
plt.title('Life Expectancy vs HDI Score')
plt.xlabel('Life Expectancy (Years)')
plt.ylabel('HDI Score')
plt.tight_layout()
plt.savefig('static/images/life_expectancy_vs_hdi.png', bbox_inches='tight', dpi=100)
plt.close()
print("Saved life_expectancy_vs_hdi.png")

# 3. Numeric Features Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df['Life expectancy'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Life Expectancy Distribution')

sns.histplot(df['Mean years of schooling'], kde=True, ax=axes[0, 1], color='olive')
axes[0, 1].set_title('Mean Years of Schooling Distribution')

sns.histplot(df['Gross national income (GNI) per capita'], kde=True, ax=axes[1, 0], color='gold')
axes[1, 0].set_title('GNI Per Capita Distribution')

sns.histplot(df['Internet users'], kde=True, ax=axes[1, 1], color='teal')
axes[1, 1].set_title('Internet Users Percentage Distribution')

plt.tight_layout()
plt.savefig('static/images/distributions.png', bbox_inches='tight', dpi=100)
plt.close()
print("Saved distributions.png")

# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_cols = ['Life expectancy', 'Mean years of schooling', 'Gross national income (GNI) per capita', 'Internet users', 'Expected years of schooling', 'HDI Score']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title('Correlation Heatmap of HDI Indicators')
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png', bbox_inches='tight', dpi=100)
plt.close()
print("Saved correlation_heatmap.png")
print("All charts generated and saved successfully!")
