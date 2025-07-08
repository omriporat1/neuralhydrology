import seaborn as sns
from IPython.display import display
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from fancyimpute import KNN


def load_and_merge_csvs(file_paths, key_column='gauge_id'):
    """Load and merge multiple CSV files on a common key."""
    dataframes = [pd.read_csv(file) for file in file_paths]
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=key_column, how='inner')
    return merged_df


'''
def analyze_features_large(df, missing_threshold=0.3, low_variance_threshold=0.01, correlation_threshold=0.95):
    """Analyze dataset with large number of features, handling non-numeric columns and large visualizations."""
    numeric_df = df.select_dtypes(include=[np.number])

    missing_values = numeric_df.isnull().mean()
    missing_columns = missing_values[missing_values > missing_threshold]

    variance = numeric_df.var()
    low_variance_columns = variance[variance < low_variance_threshold]

    correlation_matrix = numeric_df.corr()
    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_correlation_pairs.append((col1, col2))

    # Display heatmaps with adjustable size and save to file
    plt.figure(figsize=(20, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('missing_values_heatmap.png')
    plt.close()

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.1)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Prepare results DataFrame for all columns
    results = pd.DataFrame(index=df.columns)
    results['Missing_Percentage'] = df.isnull().mean()
    results['Variance'] = df.apply(lambda col: col.var() if col.dtype in [np.float64, np.int64] else np.nan)
    results['Low_Variance'] = results['Variance'] < low_variance_threshold
    results['High_Correlation'] = results.index.isin([item for sublist in high_correlation_pairs for item in sublist])

    def highlight_cells(val):
        return 'background-color: lightcoral' if val == True else ''

    display(results.style.applymap(highlight_cells, subset=['Low_Variance', 'High_Correlation']))

    return {
        "Missing Columns": missing_columns.index.tolist(),
        "Low Variance Columns": low_variance_columns.index.tolist(),
        "Variance Values": variance.to_dict(),
        "Highly Correlated Pairs": high_correlation_pairs
    }
'''


def analyze_features_plotly(df, missing_threshold=0.3, low_variance_threshold=0.01):
    """Analyze dataset with interactive Plotly heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])

    missing_values = numeric_df.isnull().mean()
    missing_columns = missing_values[missing_values > missing_threshold]

    variance = numeric_df.var()
    low_variance_columns = variance[variance < low_variance_threshold]

    correlation_matrix = numeric_df.corr()
    corr_values = correlation_matrix.abs().unstack().sort_values(kind="quicksort")
    corr_threshold = np.percentile(corr_values[corr_values < 1.0], 10)

    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corr_threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_correlation_pairs.append((col1, col2))

    filtered_corr = correlation_matrix[abs(correlation_matrix) >= corr_threshold].fillna(0)

    # Create interactive Plotly heatmap
    fig = px.imshow(filtered_corr.values,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=filtered_corr.columns,
                    y=filtered_corr.index,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    fig.update_layout(title=f'Interactive Filtered Correlation Matrix (≥ {corr_threshold:.2f})',
                      width=1000, height=1000)

    pio.write_html(fig, 'interactive_correlation_matrix.html')
    fig.show()

    results = pd.DataFrame(index=df.columns)
    results['Missing_Percentage'] = df.isnull().mean()
    results['Variance'] = df.apply(lambda col: col.var() if col.dtype in [np.float64, np.int64] else np.nan)
    results['Low_Variance'] = results['Variance'] < low_variance_threshold
    results['High_Correlation'] = results.index.isin([item for sublist in high_correlation_pairs for item in sublist])

    def highlight_cells(val):
        return 'background-color: lightcoral' if val == True else ''

    display(results.style.map(highlight_cells, subset=['Low_Variance', 'High_Correlation']))

    return {
        "Correlation Threshold": corr_threshold,
        "Missing Columns": missing_columns.index.tolist(),
        "Low Variance Columns": low_variance_columns.index.tolist(),
        "Variance Values": variance.to_dict(),
        "Highly Correlated Pairs": high_correlation_pairs
    }


def analyze_features_with_threshold(df, missing_threshold=0.3, low_variance_threshold=0.01):
    """Analyze dataset with large number of features, using correlation threshold based on the 70th percentile."""
    numeric_df = df.select_dtypes(include=[np.number])

    missing_values = numeric_df.isnull().mean()
    missing_columns = missing_values[missing_values > missing_threshold]

    variance = numeric_df.var()

    low_variance_columns = variance[variance < low_variance_threshold]

    correlation_matrix = numeric_df.corr()

    # Calculate 70th percentile threshold
    corr_values = correlation_matrix.abs().unstack().sort_values(kind="quicksort")
    corr_threshold = np.percentile(corr_values[corr_values < 1.0], 3)  # exclude perfect correlation of 1

    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corr_threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_correlation_pairs.append((col1, col2))

    # Filter correlation matrix
    filtered_corr = correlation_matrix[abs(correlation_matrix) >= corr_threshold].fillna(0)

    # Display heatmaps and save
    plt.figure(figsize=(20, 20))
    sns.heatmap(filtered_corr, cmap='coolwarm', linewidths=0.1, annot=True, fmt=".2f")
    plt.title(f'Filtered Correlation Matrix (≥ {corr_threshold:.2f})')
    plt.savefig('filtered_correlation_matrix.png')
    plt.close()

    results = pd.DataFrame(index=df.columns)
    results['Missing_Percentage'] = df.isnull().mean()
    results['Variance'] = df.apply(lambda col: col.var() if col.dtype in [np.float64, np.int64] else np.nan)
    results['Low_Variance'] = results['Variance'] < low_variance_threshold
    results['High_Correlation'] = results.index.isin([item for sublist in high_correlation_pairs for item in sublist])

    def highlight_cells(val):
        return 'background-color: lightcoral' if val == True else ''

    display(results.style.map(highlight_cells, subset=['Low_Variance', 'High_Correlation']))

    return {
        "Correlation Threshold": corr_threshold,
        "Missing Columns": missing_columns.index.tolist(),
        "Low Variance Columns": low_variance_columns.index.tolist(),
        "Variance Values": variance.to_dict(),
        "Highly Correlated Pairs": high_correlation_pairs
    }


def impute_clean_and_analyze(df, k=5):
    """Impute missing values using KNN, drop zero-variance columns, and perform similarity analysis."""
    numeric_df = df.select_dtypes(include=[np.number])

    # KNN Imputation
    imputed_data = KNN(k=k).fit_transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)

    # Drop zero-variance columns
    non_zero_var_df = imputed_df.loc[:, imputed_df.var() > 0]

    # Remove any remaining NaNs or infinite values
    clean_df = non_zero_var_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

    # Cosine Similarity
    cos_sim_matrix = cosine_similarity(clean_df.T)
    cos_sim_df = pd.DataFrame(cos_sim_matrix, index=clean_df.columns, columns=clean_df.columns)

    # Plot Cosine Similarity Heatmap using Plotly
    fig_cos = px.imshow(cos_sim_df.values,
                        labels=dict(x="Features", y="Features", color="Cosine Similarity"),
                        x=cos_sim_df.columns,
                        y=cos_sim_df.index,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)
    fig_cos.update_layout(title='Interactive Cosine Similarity Heatmap with KNN Imputation', width=1000, height=1000)
    pio.write_html(fig_cos, 'interactive_cosine_similarity_knn_cleaned.html')
    fig_cos.show()

    # Hierarchical Clustering
    plt.figure(figsize=(15, 8))
    linkage_matrix = linkage(pdist(clean_df.T, metric='cosine'), method='ward')
    dendrogram(linkage_matrix, labels=clean_df.columns, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram (Cosine Distance with KNN Imputation)")
    plt.xlabel("Features")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig('hierarchical_clustering_dendrogram_knn_cleaned.png')
    plt.show()

    return cos_sim_df


def correlation_vector_similarity(df, k=5):
    """Compute similarity between correlation vectors of features using KNN imputation."""
    numeric_df = df.select_dtypes(include=[np.number])

    # KNN Imputation
    imputed_data = KNN(k=k).fit_transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)

    # Drop zero-variance columns
    non_zero_var_df = imputed_df.loc[:, imputed_df.var() > 0]

    # Compute correlation matrix
    corr_matrix = non_zero_var_df.corr()

    # Compute cosine similarity between correlation vectors
    cos_sim_matrix = cosine_similarity(corr_matrix.T)
    cos_sim_df = pd.DataFrame(cos_sim_matrix, index=non_zero_var_df.columns, columns=non_zero_var_df.columns)

    # Interactive Plotly Heatmap
    fig_cos = px.imshow(cos_sim_df.values,
                        labels=dict(x="Features", y="Features", color="Cosine Similarity"),
                        x=cos_sim_df.columns,
                        y=cos_sim_df.index,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)
    fig_cos.update_layout(title='Interactive Cosine Similarity of Correlation Vectors', width=1000, height=1000)
    pio.write_html(fig_cos, 'interactive_correlation_vector_similarity.html')
    fig_cos.show()

    # Hierarchical Clustering on correlation vectors
    plt.figure(figsize=(15, 8))
    linkage_matrix = linkage(pdist(corr_matrix.T, metric='cosine'), method='ward')
    dendrogram(linkage_matrix, labels=non_zero_var_df.columns, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram of Correlation Vectors")
    plt.xlabel("Features")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig('hierarchical_clustering_correlation_vectors.png')
    plt.show()

    return cos_sim_df


def coefficient_of_variation_table(df):
    """Calculate and display the coefficient of variation for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])

    cv = (numeric_df.std() / numeric_df.mean().abs()).sort_values(ascending=False)

    cv_df = pd.DataFrame(cv, columns=['Coefficient_of_Variation'])
    display(cv_df.style.format("{:.4f}"))

    return cv_df


def main():

    # Simulated file paths (replace with actual paths when running)
    file_paths = ['attributes_caravan_il.csv', 'attributes_hydroatlas_il.csv', 'attributes_other_il.csv']

    # Example merged DataFrame and analysis
    merged_df = load_and_merge_csvs(file_paths, key_column='gauge_id')



    # Identify numeric columns with zero variance
    zero_variance_columns = merged_df.select_dtypes(include=[np.number]).var() == 0

    # Print column names with zero variance
    print("Zero variance columns:", zero_variance_columns[zero_variance_columns].index.tolist())

    # Remove only numeric zero-variance columns, keep non-numeric columns
    # merged_df_1 = merged_df.drop(columns=zero_variance_columns[zero_variance_columns].index)

    # save the merged_df to a csv file
    # merged_df_1.to_csv('merged_static_attributes.csv', index=False)
    # highlight_info = analyze_features_plotly(merged_df_1)
    # highlight_info

    # similarity_results = correlation_vector_similarity(merged_df, k=5)
    # similarity_results

    cv_df = coefficient_of_variation_table(merged_df)
    cv_df.to_csv('coefficient_of_variation.csv', index=True)


    a=1

if __name__ == '__main__':
    main()
