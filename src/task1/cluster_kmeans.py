
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

STOCK_STATIC_FILE = "data/stock_static.csv"
OUTPUT_FILE = "result/task1/stock_static_cluster_kmeans.csv"
ELBO_METHOD_PLOT = "result/task1/kmeans_elbow_method.png"


class ClusterKMeans():
    def __init__(self):
        self.df = None
        self.cluster_labels = None

    def load_data(self, file_path=STOCK_STATIC_FILE):
        self.df = pd.read_csv(file_path, index_col=0)

    def elbow_method(self, k_range=range(1, 20), show_plot=True):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, sse, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.savefig(ELBO_METHOD_PLOT)
        if show_plot:
            plt.show()

    def fit_kmeans(self, optimal_k=10):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.cluster_labels = final_kmeans.fit_predict(scaled_data)

        self.df['Cluster_Index'] = self.cluster_labels
        self.df.to_csv(OUTPUT_FILE, index=True)

    def visualize_clusters(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(
            self.df.drop(columns=['Cluster_Index']))

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            pca_result[:, 2],
            c=self.cluster_labels,
            cmap='viridis',
            s=50
        )

        ax.set_title('K-means Clustering')

        plt.colorbar(scatter, label='Cluster Index')
        plt.show()


if __name__ == "__main__":
    cluster_kmeans = ClusterKMeans()
    cluster_kmeans.load_data()
    cluster_kmeans.elbow_method(k_range=range(1, 20), show_plot=True)
    optimal_k = 10  # 根据肘部法则选择最佳K值
    cluster_kmeans.fit_kmeans(optimal_k=optimal_k)
    cluster_kmeans.visualize_clusters()
