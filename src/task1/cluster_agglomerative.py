import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

STOCK_STATIC_FILE = "data/stock_static.csv"
OUTPUT_FILE = "result/task1/stock_static_cluster_agglomerative.csv"
SILHOUETTE_METHOD_PLOT = "result/task1/agglomerative_silhouette_method.png"


class ClusterAgglomerative():
    def __init__(self):
        self.df = None
        self.cluster_labels = None

    def load_data(self, file_path=STOCK_STATIC_FILE):
        self.df = pd.read_csv(file_path, index_col=0)

    def silhouette_method(self, k_range=range(2, 20), show_plot=True):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        silhouette_scores = []
        for k in k_range:
            hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = hc.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method for Optimal K')
        plt.savefig(SILHOUETTE_METHOD_PLOT)
        if show_plot:
            plt.show()

    def fit_agglomerative(self, optimal_k=10):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        final_hc = AgglomerativeClustering(
            n_clusters=optimal_k, linkage='ward')
        self.cluster_labels = final_hc.fit_predict(scaled_data)

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

        ax.set_title('Agglomerative Clustering')

        plt.colorbar(scatter, label='Cluster Index')
        plt.show()


if __name__ == "__main__":
    cluster_agglomerative = ClusterAgglomerative()
    cluster_agglomerative.load_data()
    cluster_agglomerative.silhouette_method(
        k_range=range(2, 20), show_plot=True)
    optimal_k = 9  # 根据轮廓系数分析结果选择最佳K值
    cluster_agglomerative.fit_agglomerative(optimal_k=optimal_k)
    cluster_agglomerative.visualize_clusters()
