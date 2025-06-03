from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Iris dataset
data = load_iris()
X = data.data[:, :2]  # Use first two features for easy 2D plotting

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Plot the clustered data
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Crisp Partitioning of Iris Data (K-Means)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
