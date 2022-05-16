"""
Incremental agglomerative clustering, given old clusters, maps new data to old clusters
and creates new clusters for unmapped records.
"""

import re
from argparse import ArgumentParser
from ast import literal_eval

import dask.dataframe as ddf
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from clustering.preprocess import cleanse, bigram_features

NFEATURES = 1500
THRESH1 = 0.6



def cluster_only_new(new_samples: pd.DataFrame, thresh1: float, nfeatures: int) -> pd.DataFrame:
    """
    create new clusters if only new_samples are available.
    """
    vectorizer = TfidfVectorizer(max_features=nfeatures, tokenizer=lambda x: x.split(' '))
    Xnew = vectorizer.fit_transform(new_samples['spaced_tokens'])

    Xnew[Xnew==0] = 0.00001

    clusters_agg = AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=thresh1, linkage='average')\
                    .fit_predict(Xnew.todense())

    new_samples['class_prd'] = clusters_agg.astype(int)
    return new_samples

def cluster_agg_new(old_samples: pd.DataFrame, new_samples: pd.DataFrame, thresh1: float, nfeatures: int):
    """
    create new clusters if both old and new samples are available. old samples and clusters affect the 
    vectorizer and starting cluster index for new samples.
    """

    full = pd.concat((old_samples, new_samples))

    vectorizer = TfidfVectorizer(max_features=nfeatures, tokenizer=lambda x: x.split(' '))
    tfidf_mat = vectorizer.fit_transform(full['spaced_tokens'])
    X = tfidf_mat.todense()
    X[X==0]=0.00001

    Xnew = vectorizer.transform(new_samples['spaced_tokens'])
    Xnew[Xnew==0] = 0.00001

    clusters_agg = AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=thresh1, linkage='average')\
                    .fit_predict(Xnew.todense())

    unq_topics = old_samples['class_prd'].unique()
    clusters_agg = clusters_agg+max(unq_topics, default=-1)+1

    new_samples['class_prd'] = clusters_agg.astype(int)
    return vectorizer, Xnew, new_samples
    
def map_new_to_old(paired_distances: np.array, old_samples: pd.DataFrame, new_samples: pd.DataFrame, thresh1: float) -> pd.Series:
    """
    create a mapping series to map new clusters to old ones based on distance between the clusters.
    """
    n_old = len(old_samples)
    n_new = len(new_samples)
    data_long = pd.DataFrame(paired_distances.flatten(),index=pd.MultiIndex.from_product(
        [range(n_old, n_old+n_new), range(n_old)],names=['New_Ind', 'Old_Ind']),
        columns=['dist'])
    data_lr = data_long.reset_index()
    data_lr['Old_Cls'] = data_lr['Old_Ind'].map(old_samples['class_prd'])
    data_lr['New_Cls'] = data_lr['New_Ind'].map(new_samples['class_prd'])
    
    mean_dist_new_old = data_lr.pivot_table(values='dist', index='New_Cls', columns='Old_Cls', aggfunc=np.mean)
    min_dists = pd.DataFrame(mean_dist_new_old.idxmin(axis=1), columns=['min_dist_idx'])
    min_dists['index'] = min_dists.index
    min_dists['min_val'] = min_dists.apply(lambda x: mean_dist_new_old.loc[x['index'], x['min_dist_idx']], axis=1)
    min_dists['min_val'] = (min_dists['min_val'] <= thresh1).replace({False:None})
    min_dists.dropna(subset=['min_val'], inplace=True)
    mapper_new_old = min_dists['min_dist_idx']
    return mapper_new_old

def find_clusters(new_samples: pd.DataFrame, thresh1:float=THRESH1, old_samples:pd.DataFrame=None,
                  nfeatures:int=NFEATURES) -> pd.DataFrame:
    """
    This is the main function to perform incremental agglomerative clustering.
    
    Arguments
    ---------
    new_samples: DataFrame with input samples to be clustered, if preprocess == False, expects 'tokens' column
                 else expects 'content' column.
    old_samples: expected columns: ['spaced_tokens', 'class_prd'], new clusters are created if old_samples are not available
    thresh1: The distance threshold above which, clusters will not be merged.
    nfeatures: number of features to consider for tf-idf vectorizer.
    """
    print(f"threshold: {thresh1}, nfeatures: {nfeatures}")
    if 'tokens' not in new_samples.columns:
        new_samples['content_mod'] = new_samples['content'].apply(lambda x: re.sub('''[@#'"]''', '', x))
        new_samples['content_mod'] = new_samples['content_mod'].apply(lambda x: re.sub('http(s)?://[^\s]+', '', x))
        new_samples['tokens'] = new_samples['content_mod'].apply(cleanse)
        new_samples['content_mod'] = new_samples['tokens'].apply(lambda x: ' '.join(x))
    else:
        if not isinstance(new_samples['tokens'].values[0], list):
            new_samples.update(new_samples['tokens'].apply(literal_eval))
    
    no_topic_samples = new_samples[new_samples['tokens'].str.len()==0]
    new_samples=new_samples[new_samples['tokens'].str.len()>0]
    
    common_texts = new_samples['tokens'].values    
    common_texts = bigram_features(common_texts)      # add bigram features
    new_samples['tokens'] = common_texts
    new_samples['spaced_tokens'] = new_samples['tokens'].str.join(' ')
    
    # if old topics not available, create new topics only
    if type(old_samples) != pd.DataFrame:
        new_samples = cluster_only_new(new_samples, thresh1=thresh1, nfeatures=nfeatures)
        
    elif len(old_samples) == 0:
        new_samples = cluster_only_new(new_samples, thresh1=thresh1, nfeatures=nfeatures)

    else:    
        # create vectorizer with new and old samples, create new clusters
        old_samples = old_samples[old_samples['class_prd']!=-1]
        old_samples.reset_index(inplace=True, drop=True)
        new_samples['old_index'] = new_samples.index
        new_samples['index'] = range(len(old_samples), len(old_samples)+len(new_samples))
        new_samples.set_index('index', inplace=True, drop=True)
        vectorizer, Xnew, new_samples = cluster_agg_new(old_samples=old_samples, new_samples=new_samples,
                                                    thresh1=thresh1, nfeatures=nfeatures)
        Xold = vectorizer.transform(old_samples['spaced_tokens']) 
        Xold[Xold==0] = 0.00001
        # distances between each of the new and old samples 
        paired_distances = pairwise_distances(Xnew, Xold, metric='cosine')
        # map new clusters to old ones based on distances
        mapper_new_old = map_new_to_old(paired_distances=paired_distances, old_samples=old_samples, new_samples=new_samples, thresh1=thresh1)
        new_samples['class_prd'] = new_samples['class_prd'].apply(lambda x: mapper_new_old[x] if x in mapper_new_old.index else x)
    if len(no_topic_samples)>0:
        no_topic_samples['class_prd']=-1
        new_samples = new_samples.append(no_topic_samples)
    return new_samples


if __name__=='__main__':
    caller=ArgumentParser()
    caller.add_argument('newSamplesPath', help="Path to samples to be clustered")
    caller.add_argument('outputPath', help="Path to save output")
    caller.add_argument('--oldSamplesPath', help="Path to samples already clustered if any")
    caller.add_argument('--threshold', help="distance threshold, samples with distance above this will not be clustered together, default=0.6", type=float)
    caller.add_argument('--numFeatures', help="number of features to be considered for tf-idf vectorizer, default=1500")
    
    z = caller.parse_args()
    new_samples = pd.read_csv(z.newSamplesPath)
    old_samples = None
    thresh1 = THRESH1
    nfeatures = NFEATURES
    pre_process = False
    if z.oldSamplesPath:
        old_samples = pd.read_csv(z.oldSamplesPath)
    if z.threshold:
        thresh1 = z.threshold
    if z.numFeatures:
        nfeatures = z.numFeatures
    out = find_clusters(new_samples, thresh1=thresh1, old_samples=old_samples, nfeatures=nfeatures)
    out.to_csv(z.outputPath)
    