from benchmarks.plotting.draw_benchmark_plots import print_mRecall_pointplot
import benchmarks.plotting.benchmark_metrics as benchmark_metrics


# Set var for benchmarks:
datasets = ['municipios', 'MNIST', 'NYtimes', 'GLOVE']
#datasets = ['MNIST']
distances = ['manhattan', 'euclidean', 'chebyshev', 'cosine']
methods = ['FLANN', 'PYNN', 'GDASC']
baseline = 'Exact'
knn = [5, 10, 15]

mask_algorithm = 'kmeans'
mask_implementation = 'kclust'

gmask_algorithm = 'kmedoids'
gmask_implementation = 'fastkmedoids'


if __name__ == '__main__':

    # Show results on a graph

    # Obtain recall of each (k=5,10,15) experiment
    #recalls = benchmark_metrics.get_recall(datasets, distances, methods, knn, gmask_algorithm, gmask_implementation, baseline)

    # Print recall graph
    # print_compare_recall_graph(recalls)
    # print_compare_recall_boxplots(recalls)
    # print_recall_heatmap(datasets, distances, methods, knn, recalls)

    # Obtain mean Recall of all (k=5,10,15) experiments
    mRecall = benchmark_metrics.get_mRecall(datasets, distances, methods, knn, gmask_algorithm, gmask_implementation, baseline)

    # Print mRecall graph
    # print_mRecall_barplot(datasets, distances, methods, mRecall) #barplot
    print_mRecall_pointplot(datasets, distances, methods, mRecall) #pointplot

exit(0)
