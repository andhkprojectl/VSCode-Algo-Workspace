import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_COL = 'rt10'         # Analyzing rt10 > 0 condition
TARGET_THRESHOLD = 0        # Threshold for rt10
# Condition evaluated: df[TARGET_COL] > TARGET_THRESHOLD
# ==========================================

# File paths
input_file = r"c:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_8001_20260204_NVDA\csvExcel\NVDA_explore_fm_amibroker_trim.csv"
output_dir = r"c:\Project\ProjectLife\VSCode Algo Workspace\VS_8000_Strategy\VS_8001_20260204_NVDA\1 Prompt"
output_plot = os.path.join(output_dir, "feature_analysis_plot.png")

# 1. Load the data (tab-separated Amibroker export)
df = pd.read_csv(input_file, sep='\t')

# 2. Create binary target: 1 if rt10 > 0, else 0
df['target'] = (df[TARGET_COL] > TARGET_THRESHOLD).astype(int)

print(f"Analyzing features related to condition: {TARGET_COL} > {TARGET_THRESHOLD}")
print(f"Total rows: {len(df)}, Positive cases (rt10>0): {df['target'].sum()}, Negative: {(df['target']==0).sum()}\n")
print("-" * 60)

# 3. Isolate numeric features (exclude metadata + target columns)
exclude_cols = ['Symbol', 'Trade', 'Date', 'target', TARGET_COL, 'rt1']
# Note: rt1 is excluded because it is the 1-bar return and trivially correlated with rt10
numeric_features = df.select_dtypes(include=[np.number]).drop(
    columns=[col for col in exclude_cols if col in df.columns],
    errors='ignore'
)

print(f"Analyzing {len(numeric_features.columns)} numeric features...\n")

# --- METHOD 1: Pearson Correlation (Linear Relationship) ---
correlations = numeric_features.corrwith(df['target']).abs().sort_values(ascending=False)
print("=== Top 15 Features by Absolute Pearson Correlation ===")
print(correlations.head(15).to_string())
print("\n" + "-" * 60)

# --- METHOD 2: Mutual Information (Non-Linear Relationship) ---
mi_scores = mutual_info_classif(numeric_features, df['target'], random_state=42)
mi_series = pd.Series(mi_scores, index=numeric_features.columns).sort_values(ascending=False)
print("=== Top 15 Features by Mutual Information Score ===")
print(mi_series.head(15).to_string())
print("\n" + "-" * 60)

# --- METHOD 3: Random Forest Feature Importance (Complex Combinations) ---
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(numeric_features, df['target'])
rf_importance = pd.Series(rf.feature_importances_, index=numeric_features.columns).sort_values(ascending=False)
print("=== Top 15 Features by Random Forest Importance ===")
print(rf_importance.head(15).to_string())
print("\n" + "-" * 60)

# --- METHOD 4: Mean comparison (rt10>0 vs rt10=0) ---
print("=== Mean Value Comparison: rt10>0 vs rt10=0 (Top 10 correlated) ===")
top10 = correlations.head(10).index.tolist()
for feat in top10:
    pos_mean = df[df['target'] == 1][feat].mean()
    neg_mean = df[df['target'] == 0][feat].mean()
    diff = pos_mean - neg_mean
    print(f"  {feat:40s}  rt10>0: {pos_mean:10.4f}  rt10=0: {neg_mean:10.4f}  diff: {diff:+10.4f}")
print("\n" + "-" * 60)

# --- FEATURE COMBINATION ANALYSIS ---
print("\n" + "=" * 60)
print("=== FEATURE COMBINATION ANALYSIS (Pairwise Interactions) ===")
print("=" * 60)

top5 = correlations.head(5).index.tolist()
print(f"\nTop 5 features for combination analysis: {top5}")

print("\n--- Pairwise Interaction Correlations ---")
pair_results = []
for feat1, feat2 in combinations(top5, 2):
    interaction = df[feat1] * df[feat2]
    corr = abs(interaction.corr(df['target']))
    pair_results.append((f"{feat1} × {feat2}", corr))

pair_results.sort(key=lambda x: x[1], reverse=True)
for name, corr in pair_results:
    print(f"  {name:60s}  |corr| = {corr:.4f}")

# --- TRIPLE COMBINATION ANALYSIS ---
print("\n--- Triple Interaction Correlations ---")
triple_results = []
for feat1, feat2, feat3 in combinations(top5, 3):
    interaction = df[feat1] * df[feat2] * df[feat3]
    corr = abs(interaction.corr(df['target']))
    triple_results.append((f"{feat1} × {feat2} × {feat3}", corr))

triple_results.sort(key=lambda x: x[1], reverse=True)
for name, corr in triple_results:
    print(f"  {name:70s}  |corr| = {corr:.4f}")

# --- VISUALIZATION ---
print("\nGenerating visualizations...")
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 18))

# Plot 1: Correlation bar chart
ax1 = fig.add_subplot(4, 1, 1)
sns.barplot(x=correlations.head(15).values, y=correlations.head(15).index, ax=ax1, palette='viridis')
ax1.set_title(f'Top 15 Features by |Pearson Correlation| with (rt10 > 0)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Absolute Correlation')

# Plot 2: Mutual Information bar chart
ax2 = fig.add_subplot(4, 1, 2)
sns.barplot(x=mi_series.head(15).values, y=mi_series.head(15).index, ax=ax2, palette='magma')
ax2.set_title('Top 15 Features by Mutual Information Score', fontsize=13, fontweight='bold')
ax2.set_xlabel('Mutual Information')

# Plot 3: Random Forest Importance bar chart
ax3 = fig.add_subplot(4, 1, 3)
sns.barplot(x=rf_importance.head(15).values, y=rf_importance.head(15).index, ax=ax3, palette='crest')
ax3.set_title('Top 15 Features by Random Forest Importance', fontsize=13, fontweight='bold')
ax3.set_xlabel('Feature Importance')

# Plot 4: Heatmap of top features correlation matrix
ax4 = fig.add_subplot(4, 1, 4)
top_heatmap = correlations.head(10).index.tolist()
corr_matrix = df[top_heatmap + ['target']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax4,
            linewidths=0.5, annot_kws={'size': 7})
ax4.set_title('Correlation Heatmap: Top 10 Features + Target (rt10>0)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_plot}")
plt.show()

# --- DETAILED ROW INSPECTION ---
print(f"\n{'='*60}")
print(f"=== ALL ROWS WHERE rt10 > 0 (showing top 5 correlated features) ===")
print(f"{'='*60}")
positive_df = df[df['target'] == 1]
cols_to_show = ['Date', 'rt10'] + top5
print(positive_df[cols_to_show].to_string(index=False))

print(f"\n=== ALL ROWS WHERE rt10 = 0 (showing top 5 correlated features) ===")
negative_df = df[df['target'] == 0]
print(negative_df[cols_to_show].to_string(index=False))

# --- SUMMARY ---
print(f"\n{'='*60}")
print("=== SUMMARY ===")
print(f"{'='*60}")
print(f"Target: rt10 > 0")
print(f"Total bars: {len(df)}")
print(f"Bars with rt10 > 0: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
print(f"Bars with rt10 = 0: {(df['target']==0).sum()} ({(1-df['target'].mean())*100:.1f}%)")
print(f"\nTop 3 single features (correlation): {correlations.head(3).index.tolist()}")
print(f"Top 3 single features (mutual info):  {mi_series.head(3).index.tolist()}")
print(f"Top 3 single features (random forest): {rf_importance.head(3).index.tolist()}")
if pair_results:
    print(f"Best feature pair: {pair_results[0][0]} (|corr|={pair_results[0][1]:.4f})")
if triple_results:
    print(f"Best feature triple: {triple_results[0][0]} (|corr|={triple_results[0][1]:.4f})")