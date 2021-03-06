# Clustering
This repository contains implementation of new clustering methods and utilities based on recent research papers.

# 1 Incremental agglomerative clustering
Incremental agglomerative clustering, given old clusters, maps new data to old clusters and creates new clusters for unmapped records. It is a bottom-up approach, meaning it assumes all the data points belong to separate clusters initially. Then it recursively merges the cluster pairs which have minimum distance between them. This kind of approach is useful when we are dealing with temporal text data and need to cluster it incrementally in time. For example, news, social media posts, chats etc. which keep on increasing with time and there is no endpoint to wait for before doing the analysis. This implementation is based on the following paper:

* X. Dai, Q. Chen, X. Wang and J. Xu, "Online topic detection and tracking of financial news based on hierarchical clustering," 2010 International Conference on Machine Learning and Cybernetics, 2010, pp. 3341-3346, doi: 10.1109/ICMLC.2010.5580677.


# 2 Compare Clusters
## map and compare clusters to ground truth clusters using f-measure as the metric.
It is a useful method to measure how much the clustering results of two different algorithms match and which clusters from one result map to which clusters from othwr result. It can also be used to do this matching between predicted clustering result and ground truth clusters if available. The algorithm used here is available in:

* Wagner, Silke, and Dorothea Wagner. Comparing clusterings: an overview. Karlsruhe: Universität Karlsruhe, Fakultät für Informatik, 2007.

The basic algorithm is as follows:

Lets assume ground truth has M number of clusters and clustering result has N number of clusters.

For each m<sup>th</sup> cluster in ground truth, calculate f-measure with every cluster in clustering result. This f-measure indicates how good the cluster C<sub>n</sub> describes the cluster C<sub>m</sub>.

I<sub>mn</sub> → Intersection of elements in m<sup>th</sup> cluster in ground truth and n<sup>th</sup> cluster in predicted clusters.

|C<sub>m</sub>| = number of elements in m<sup>th</sup> cluster

precision p = I<sub>mn</sub>/|C<sub>n</sub>|,            recall r = I<sub>mn</sub>/|C<sub>m</sub>|

F-measure of mth and nth cluster fmn = 2.r.p/(r+p) = 2.I<sub>mn</sub>/(|C<sub>m</sub>|+|C<sub>n</sub>|)

2. Create a matrix with cluster labels in ground truth  as row index, cluster labels in results as column index and f-measures of clusters as values.

3. Identify the cluster pair with maximum f-measure, assume that these clusters are mapped and store these mappings and corresponding f-measures, remove the row and column corresponding to these clusters. Repeat this until we get empty matrix.

4. Overall f-measure is the average of f-measure corresponding to each cluster map identified in previous step. 


**Please check readme_help_example.ipynb for more details.**

For setup, install required packages listed in requirements.txt
