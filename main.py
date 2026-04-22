from src.task1.correlation_coefficient import Corr
from src.task1.feature_analysis import FeatureAnalysis
from src.task1.cluster_kmeans import ClusterKMeans
from src.task1.cluster_agglomerative import ClusterAgglomerative

from src.task2.average_return import AverageReturn
from src.task2.variance_interpret import VarianceInterpretation


if __name__ == "__main__":

    # 1.1 特征分析
    analysis = FeatureAnalysis()
    analysis.process_and_save()
    analysis.print_results(top_n=10)

    # 1.2 相关系数分析
    corr = Corr()
    corr.load_data()
    corr.compute_similarity(show_heatmap=False)
    corr.print_top_pairs()
    corr.save_results()

    # 1.3.1 K-Means聚类
    kmeans = ClusterKMeans()
    kmeans.load_data()
    # 根据肘部法则选择最佳K值
    kmeans.elbow_method(k_range=range(1, 20), show_plot=False)
    kmeans.fit_kmeans(optimal_k=10)
    # kmeans.visualize_clusters() # 可视化聚类结果

    # 1.3.2 层次聚类
    agglomerative = ClusterAgglomerative()
    agglomerative.load_data()
    # 根据轮廓系数分析结果选择最佳K值
    agglomerative.silhouette_method(k_range=range(
        2, 20), show_plot=False)
    agglomerative.fit_agglomerative(optimal_k=9)
    # agglomerative.visualize_clusters() # 可视化聚类结果

    # 2.1 平均收益率计算
    avg_return = AverageReturn()
    avg_return.compute_average_return()
    avg_return.plot_average_return(show_plot=False)
    avg_return.save_average_return(avg_return.index_returns)

    # 2.2 方差解释率分析
    variance_interp = VarianceInterpretation()
    variance_interp.load_returns_from_config()
    variance_interp.compute_and_plot()
