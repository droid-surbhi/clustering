# Clustering
This repository contains implementation of new clustering methods and utilities based on recent research papers.

# Incremental agglomerative clustering
Incremental agglomerative clustering, given old clusters, maps new data to old clusters and creates new clusters for unmapped records. It is a bottom-up approach, meaning it assumes all the data points belong to separate clusters initially. Then it recursively merges the cluster pairs which have minimum distance between them. This kind of approach is useful when we are dealing with temporal text data and need to cluster it incrementally in time. For example, news, social media posts, chats etc. which keep on increasing with time and there is no endpoint to wait for before doing the analysis. This implementation is based on the following paper:

* X. Dai, Q. Chen, X. Wang and J. Xu, "Online topic detection and tracking of financial news based on hierarchical clustering," 2010 International Conference on Machine Learning and Cybernetics, 2010, pp. 3341-3346, doi: 10.1109/ICMLC.2010.5580677.


**Please check readme_help_example.ipynb for more details.**

For setup, install required packages listed in requirements.txt
