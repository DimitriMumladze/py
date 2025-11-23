# ============================================================================
# MALL CUSTOMERS CLUSTERING ANALYSIS
# ============================================================================
# This script demonstrates unsupervised learning (clustering) techniques:
# 1. K-Means Clustering with visualization
# 2. Agglomerative (Hierarchical) Clustering with Dendrogram
# 3. Silhouette Score for cluster quality evaluation
# Dataset: Mall_Customers.csv (customer segmentation data)
# ============================================================================

# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.cluster import KMeans, AgglomerativeClustering  # Clustering algorithms
from scipy.cluster.hierarchy import linkage, dendrogram  # For hierarchical clustering visualization
from sklearn.metrics import silhouette_score  # To evaluate cluster quality
import matplotlib.pyplot as plt  # For creating visualizations

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

# Load the Mall Customers dataset from URL
data = pd.read_csv("https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv")

# Display basic information about the dataset
print("=" * 70)
print("MALL CUSTOMERS DATASET")
print("=" * 70)
print("Original Data:")
print(data.head())
print(f"\nDataset Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\n")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Drop CustomerID and Age columns as they are not needed for this analysis
# axis=1 means drop columns (not rows)
# inplace=True modifies the dataframe directly
data.drop(['CustomerID', 'Age'], axis=1, inplace=True)

print("Data after preprocessing:")
print(data.head())
print("\n")

# ============================================================================
# EXPLORATORY DATA VISUALIZATION
# ============================================================================

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Create a scatter plot to visualize the relationship between Income and Spending
plt.figure(figsize=(10, 8))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')  # X-axis label
plt.ylabel('Spending Score (1-100)')  # Y-axis label
plt.title('Customer Distribution: Income vs Spending Score')
plt.grid(True, alpha=0.3)  # Add subtle grid
plt.show()

# ============================================================================
# K-MEANS CLUSTERING (5 CLUSTERS)
# ============================================================================

print("=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

# Create a K-Means model with 5 clusters
# n_clusters=5 means we want to divide customers into 5 groups
clustering_kmeans = KMeans(n_clusters=5)

# Fit the model and predict cluster labels for each customer
# We use only 'Annual Income' and 'Spending Score' columns
labels_kmeans = clustering_kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])

# Visualize the K-Means clustering results
plt.figure(figsize=(12, 8))

# Plot each customer colored by their cluster
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=labels_kmeans, cmap='viridis')

# Plot cluster centers as blue dots
# cluster_centers_[:,0] gets x-coordinates (income)
# cluster_centers_[:,1] gets y-coordinates (spending score)
plt.scatter(clustering_kmeans.cluster_centers_[:, 0], 
            clustering_kmeans.cluster_centers_[:, 1], 
            c='blue', s=200, marker='X', edgecolors='black', linewidths=2,
            label='Cluster Centers')

# Add gender labels to each point for additional insight
for i, gender in enumerate(data['Gender']):
    plt.annotate(text=gender, 
                xy=(data['Annual Income (k$)'][i], data['Spending Score (1-100)'][i]),
                fontsize=6, alpha=0.6)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering: Customer Segmentation (5 Clusters)')
plt.colorbar(label='Cluster')  # Add color bar to show cluster numbers
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Display inertia (sum of squared distances to nearest cluster center)
# Lower inertia means tighter clusters
print(f"K-Means Inertia: {clustering_kmeans.inertia_:.2f}")
print("\n")

# ============================================================================
# K-MEANS WITH 3 CLUSTERS (Age vs Spending Score)
# ============================================================================

print("=" * 70)
print("K-MEANS CLUSTERING - ALTERNATIVE (Age vs Spending)")
print("=" * 70)

# Reload data for alternative clustering approach
customer = pd.read_csv("https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv")

# Drop different columns this time (keeping Age, dropping Income)
customer.drop(["CustomerID", "Annual Income (k$)"], axis=1, inplace=True)

# Create K-Means model with 3 clusters
model_kmeans_3 = KMeans(n_clusters=3)

# Fit and predict clusters
color_predicted = model_kmeans_3.fit_predict(customer[["Age", "Spending Score (1-100)"]])

# Visualize the results
plt.figure(figsize=(10, 8))
plt.scatter(x=customer["Age"], y=customer["Spending Score (1-100)"], c=color_predicted, cmap='plasma')

# Plot cluster centers
plt.scatter(x=model_kmeans_3.cluster_centers_[:, 0], 
            y=model_kmeans_3.cluster_centers_[:, 1], 
            c="red", s=200, marker='X', edgecolors='black', linewidths=2,
            label='Cluster Centers')

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering: Age vs Spending Score (3 Clusters)')
plt.colorbar(label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"K-Means (3 clusters) Inertia: {model_kmeans_3.inertia_:.2f}")
print("\n")

# ============================================================================
# HIERARCHICAL (AGGLOMERATIVE) CLUSTERING WITH DENDROGRAM
# ============================================================================

print("=" * 70)
print("HIERARCHICAL CLUSTERING")
print("=" * 70)

# Create linkage matrix for dendrogram
# This calculates the hierarchical clustering structure
z = linkage(customer[["Age", "Spending Score (1-100)"]])

# Plot dendrogram to visualize hierarchical clustering
plt.figure(figsize=(15, 8))
dendrogram(z)  # Create dendrogram plot
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.axhline(y=200, color='r', linestyle='--', label='Cut line for 5 clusters')  # Suggested cut line
plt.legend()
plt.show()

# Create Agglomerative Clustering model with 5 clusters
# linkage='ward' minimizes variance within clusters
model_agg = AgglomerativeClustering(n_clusters=5, linkage="ward")

# Fit and predict cluster labels
label_agg = model_agg.fit_predict(customer[["Age", "Spending Score (1-100)"]])

# Prepare data for silhouette score calculation
x = customer[["Age", "Spending Score (1-100)"]]

# Calculate silhouette score (measures cluster quality)
# Score ranges from -1 to 1, where higher is better
# Values near 0 indicate overlapping clusters
silhouette = silhouette_score(x, label_agg)

print(f"Agglomerative Clustering Silhouette Score: {silhouette:.4f}")
print("Interpretation: Higher score (close to 1) means better-defined clusters")
print("\n")

# Visualize Agglomerative Clustering results
plt.figure(figsize=(10, 8))
plt.scatter(x=customer["Age"], y=customer["Spending Score (1-100)"], c=label_agg, cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Agglomerative Clustering: Age vs Spending Score (5 Clusters)')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("CLUSTERING ANALYSIS SUMMARY")
print("=" * 70)
print(f"K-Means (5 clusters) Inertia: {clustering_kmeans.inertia_:.2f}")
print(f"K-Means (3 clusters) Inertia: {model_kmeans_3.inertia_:.2f}")
print(f"Agglomerative Clustering Silhouette Score: {silhouette:.4f}")
print("\nNote: Lower inertia = tighter clusters")
print("      Higher silhouette score = better cluster separation")
print("=" * 70)
