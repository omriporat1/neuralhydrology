from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


# Load data
df = pd.read_csv("random_search_configurations_denser.csv", index_col="job_id")

# Standardize for PCA and t-SNE
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# --- 1. 2D PCA ---
pca_2d = PCA(n_components=2)
df_pca2d = pd.DataFrame(pca_2d.fit_transform(scaled_data), columns=["PC1", "PC2"])

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", data=df_pca2d)
plt.title("2D PCA Projection of Hyperparameter Configurations")
plt.tight_layout()
plt.show()

# --- 2. 3D PCA (interactive with Plotly) ---
pca_3d = PCA(n_components=3)
df_pca3d = pd.DataFrame(pca_3d.fit_transform(scaled_data), columns=["PC1", "PC2", "PC3"])
df_pca3d["job_id"] = df.index

fig_pca = px.scatter_3d(df_pca3d, x="PC1", y="PC2", z="PC3",
                        hover_name="job_id", title="3D PCA of Hyperparameter Configurations")
fig_pca.show()

# --- 3. 2D t-SNE (interactive with Plotly) ---
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
df_tsne = pd.DataFrame(tsne.fit_transform(scaled_data), columns=["TSNE1", "TSNE2"])
df_tsne["job_id"] = df.index

fig_tsne = px.scatter(df_tsne, x="TSNE1", y="TSNE2",
                      hover_name="job_id", title="2D t-SNE of Hyperparameter Configurations")
fig_tsne.show()

# --- 4. Parallel Coordinates Plot ---
df_parallel = df.copy()
df_parallel["index"] = df_parallel.index.astype(str)

plt.figure(figsize=(20, 9))
parallel_coordinates(df_parallel, "index", colormap=plt.cm.tab20, alpha=0.5)
plt.title("Parallel Coordinates Plot of Hyperparameter Configurations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 5. Pairplot of Selected Features ---
selected_features = ["batch_size", "hidden_size", "learning_rate", "output_dropout"]
sns.pairplot(df[selected_features])
plt.suptitle("Pairplot of Selected Hyperparameters", y=1.02)
plt.tight_layout()
plt.show()

# Compute correlation matrix
corr = df.corr()

# --- 1. Heatmap of feature correlations ---
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of Hyperparameters")
plt.tight_layout()
plt.show()

# --- 2. Improved pairplot: color-coded with jitter to reduce overplotting ---
df_jittered = df.copy()
# Add small jitter to numeric columns for visibility
jitter_strength = {
    'batch_size': 10,
    'hidden_size': 10,
    'learning_rate': 0.0002,
    'output_dropout': 0.02,
    'seq_length': 3,
    'statics_embedding': 1
}

for col, jitter in jitter_strength.items():
    df_jittered[col] += np.random.uniform(-jitter, jitter, size=len(df))

# Assign arbitrary clusters for color (e.g., k-means would go here if needed)
df_jittered['group'] = pd.qcut(df['learning_rate'], q=3, labels=["low_lr", "med_lr", "high_lr"])

sns.pairplot(df_jittered, hue="group", plot_kws={"alpha": 0.8, "s": 60})
plt.suptitle("Pairplot with Jitter and Colored by Learning Rate Bins", y=1.02)
plt.tight_layout()
plt.show()

# --- 3. Optional: Enhanced parallel coordinates with grouping ---
from sklearn.preprocessing import MinMaxScaler

df_scaled = df.copy()
df_scaled['group'] = df_jittered['group']
df_scaled[['batch_size', 'hidden_size', 'learning_rate', 'output_dropout', 'seq_length', 'statics_embedding']] = \
    MinMaxScaler().fit_transform(df_scaled.drop(columns='group'))

plt.figure(figsize=(14, 6))
parallel_coordinates(df_scaled, "group", colormap=plt.cm.Set1, alpha=0.6)
plt.title("Normalized Parallel Coordinates Colored by Learning Rate Bins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_jittered = df.copy()
jitter_strength = {
    'batch_size': 10,
    'hidden_size': 10,
    'learning_rate': 0.0002,
    'output_dropout': 0.02,
    'seq_length': 3,
    'statics_embedding': 1
}
for col, jitter in jitter_strength.items():
    df_jittered[col] += np.random.uniform(-jitter, jitter, size=len(df))

# Add categorical grouping for color (based on learning_rate)
df_jittered['group'] = pd.qcut(df['learning_rate'], q=3, labels=["low_lr", "med_lr", "high_lr"])

# Plot only numeric columns
numeric_cols = ['batch_size', 'hidden_size', 'learning_rate', 'output_dropout', 'seq_length', 'statics_embedding']
sns.pairplot(df_jittered[numeric_cols + ['group']], hue="group", diag_kind="hist",
             plot_kws={"alpha": 0.8, "s": 60})
plt.suptitle("Pairplot with Jitter, Histograms, Colored by Learning Rate Bins", y=1.02)
plt.tight_layout()
plt.show()

