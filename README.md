# GDASC - General Distributed Approximate Similarity search with Clustering

![GDASC](benchmarks/figures/fig_multilayer_structure.png)

<!-- 
Finding elements from a dataset is the main task of similarity search which is typically achieved by representing the data as feature vectors in a multidimensional space, and then applying a specific similarity or dissimilarity metric to compare the query object to the elements in the dataset. Diverse data indexing techniques, collectively called access methods, have been proposed to expedite this process. In this context, two constraints are evident in the (existing) algorithms proposed regarding the issues inherent in representing datasets in high-dimensional spaces and the choice of indexing metric or method.-->
<!-- 
One of the newest approaches is Mask (Multilevel Approximate Similarity search with k-means) (Ortega et al., 2023), a novel indexing method that involves an unconventional application of the 𝑘-means partitioning algorithm (MacQueen, 1967; Lloyd, 1982) to create a multilevel index structure for approximate similarity search.-->
* This project proposes GDASC (General Distributed Approximate Similarity search with Clustering) a novel generalized algorithm for distributed approximate similarity search that accepts any arbitrary distance function. It employs clustering algorithms that induce Voronoi regions in a dataset and yield a representative element, such as k-medoids, to build a multilevel indexing structure suitable for large datasets with high dimensionality and sparsity, usually stored in distributed systems.
<!-- * This project proposes a new algorithm GDASC (General Distributed Approximate Similarity search with Clustering), a novel algorithm designed for efficient approximate similarity search.-->




<!-- 
This project proposes a new algorithm GDASC that is a generalized algorithm to solve the approximate nearest neighbours (ANN) search problem for distributed data that accepts any arbitrary distance function by employing data partitioning algorithms that induce Voronoi regions in a dataset and yield a representative element, such as k-medoids.-->

## Installation:
## Summary of features
* GDASC constructs a multilevel indexing structure, making it suitable for large, high-dimensional, and sparse datasets typically stored in distributed systems.<!-- This algorithm is adaptable with various clustering algorithms, including k-means, k-medoids, and DBSCAN..., and has already been successfully applied to k-medoids.-->
* GDASC, is an adaptable algorithm  with various clustering algorithms, such as k-means and k-medoids. Notably, it has already been successfully applied with the k-medoids algorithm, demonstrating its versatility and effectiveness in diverse clustering scenarios.

* GDASC uses k-medoids to enhance compatibility with any distance function, Unlike many similarity search algorithms that rely on k-means (usually associated with the Euclidean distance).


### Clustering Algorithms:


| Algorithm    | Parameters                                         | Implementation                                    |
|--------------|----------------------------------------------------|---------------------------------------------------|
| **K-Means**  | - `n_clusters`: Number of clusters (default: 8)    | - `scikit-learn`: `sklearn.cluster.KMeans`        |
|              | - `init`: Method for initialization (default: 'k-means++') | - `The Algorithms`: `Python.machine_learning.k_means_clust`|
|              | - `n_init`: Number of time the algorithm will run with different centroid seeds (default: 10) |                                                   |
|              | - `max_iter`: Maximum number of iterations (default: 300) |                                                   |
|              | - `tol`: Relative tolerance with regards to inertia to declare convergence (default: 1e-4) |                                                   |
| **k-medoids**| - `n_clusters`: Number of clusters (default: 8)    | - `scikit-learn-extra`: `sklearn_extra.cluster.KMedoids` |
|              | - `init`: Method for initialization (default: 'heuristic') | - `PyClustering`: `pyclustering.cluster.kmedoids` |
|              | - `max_iter`: Maximum number of iterations (default: 300) |                                                   |
|              | - `metric`: The distance metric to use (default: 'euclidean') |                                                   |


### Supported distances:

| Distance      | API         | Équation                                    |
|---------------|-------------|---------------------------------------------|
| **Euclidean** | `euclidean` | ![d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20%28x_i%20-%20y_i%29%5E2%7D) |
| **Manhattan** | `manhattan` | ![d(x, y) = \sum_{i=1}^n \|x_i - y_i\|](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20%7Cx_i%20-%20y_i%7C) |
| **Chebyshev** | `chebyshev` | ![d(x, y) = \max_i \|x_i - y_i\|](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Cmax_i%20%7Cx_i%20-%20y_i%7C) |
| **Minkowski** | `minkowski` | ![d(x, y) = \left( \sum_{i=1}^n \|x_i - y_i\|^p \right)^{\frac{1}{p}}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%20%5Cleft%28%20%5Csum_%7Bi%3D1%7D%5En%20%7Cx_i%20-%20y_i%7C%5Ep%20%5Cright%29%5E%7B%5Cfrac%7B1%7D%7Bp%7D%7D) |
| **Cosine**    | `cosine`    | ![d(x, y) = 1 - \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2} \sqrt{\sum_{i=1}^n y_i^2}}](https://latex.codecogs.com/svg.latex?d%28x%2C%20y%29%20%3D%201%20-%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5En%20x_i%20y_i%7D%7B%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20x_i%5E2%7D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%20y_i%5E2%7D%7D) |
  
    
## Datasets:
| Dataset            | Label       | N         | Dimensionality | High Sparsity | Data Type  | Download                                           |
|--------------------|-------------|-----------|----------------|---------------|------------|---------------------------------------------------|
| **Municipalities** | X_muni      | 8,130     | 2              | No            | Geospatial | [Municipalities](https://doi.org/10.5281/zenodo.12759082) |
| **MNIST**          | X_MNIST     | 69,000    | 784            | Yes           | Image      | [MNIST](https://doi.org/10.5281/zenodo.12759284)          |
| **GLOVE**          | X_GLOVE     | 1,000,000 | 100            | No            | Text       | [GLOVE](https://doi.org/10.5281/zenodo.12759356)          |
| **NYtimes**        | X_NYtimes   | 290,000   | 256            | No            | Text       | [NYtimes](https://doi.org/10.5281/zenodo.12760693)        |





## Benchmarks:

* We have consolidated the experiments for 5, 10, and 15 neighbors and calculated the average recall for these three conditions. 
* The point plot illustrates the recall of three different algorithms across various distance metrics (Manhattan, Euclidean, Chebyshev, Cosine) in approximate nearest neighbor search for each dataset.
     
  *  ### Approximate Algorithms:
  
     * __PyNNDescent__  constructs a graph-based index by connecting each data point to its approximate nearest neighbours. It iteratively improves the graph structure through neighbour descent steps, employing techniques such as random initialisation, multi-tree searching, and pruning. The algorithm also provides parameters to control the trade-off between search accuracy and computational cost.
     * __FLANN__ employs techniques such as random projection and hierarchical subdivision to construct an index structure that enables faster search operations.
 
### Dataset 1:

![municipalities](benchmarks/figures/municipios_avgRecall.png)

<!-- ### Mask
Mask (Multilevel Approximate Similarity search with k-means) (Ortega et al., 2023), a novel indexing method that involves an unconventional application of the 𝑘-means partitioning algorithm (MacQueen, 1967; Lloyd, 1982) to create a multilevel index structure for approximate similarity search. -->

### Dataset 2:
![mnist](benchmarks/figures/MNIST_avgRecall.png)

### Dataset 3:
![glove](benchmarks/figures/GLOVE_avgRecall.png)

### Dataset 4:

![NYtimes](benchmarks/figures/NYtimes_avgRecall.png)



  



