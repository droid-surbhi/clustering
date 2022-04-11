"""
map and compare clusters to ground truth clusters
"""

import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser


def preprocess(text: str) -> str:
    """
    same set of words in different order or having extra special characters should be identified as same topic
    """
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)
    text = re.sub('[^a-z]',' ',text)
    text = re.sub('\s+', ' ', text)
    text2 = " ".join(sorted(list(set(text.split(" ")))))
    return text2

def join_count(df: pd.DataFrame, source: str, dest_name: str) -> pd.DataFrame:
    counts = df[source].value_counts()
    counts.name = dest_name
    df = df.set_index(source, drop=False).join(counts).reset_index(drop=True)
    return df

def eval_fmeasure(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculate f-measure per cluster
    """
    count_agg = df.groupby(['class_gt', 'class_prd']).apply(lambda x: x['gt_class_count'].values[0] + x['prd_class_count'].values[0])
    count_agg = pd.DataFrame(count_agg, columns=['#total_elements'])
    df_possible = df.set_index('class_gt')
    predicted_sum_count = df_possible.groupby(level=0).apply(lambda x:x['class_prd'].value_counts())
    predicted_sum_count.index.names = ['class_gt', 'class_prd']
    predicted_sum_count.name = 'gt.prd'
    count_agg = count_agg.join(predicted_sum_count)
    count_agg['fmeasure']=count_agg.apply(lambda x: (2*x['gt.prd'])/x['#total_elements'], axis=1)
    return count_agg

def map_rough_clusters(unmatched: pd.DataFrame()) -> dict:
    """
    match clusters with maximum f-measure
    """
    unm_df = unmatched.reset_index()
    df_pvt = pd.pivot(unm_df, index='class_gt', columns='class_prd', values='fmeasure').fillna(0)
    cols = df_pvt.columns
    idxs = df_pvt.index
    rough_match = {}

    while df_pvt.shape[0]>0 and df_pvt.shape[1]>0:
        cl_mx = df_pvt.max().idxmax()
        rw_mx = df_pvt.idxmax().loc[cl_mx]
        rough_match[rw_mx] = (cl_mx, df_pvt.loc[rw_mx, cl_mx])
        df_pvt.drop(index=rw_mx, inplace=True)
        df_pvt.drop(columns=cl_mx, inplace=True)
    return rough_match

def cluster_measure(df: pd.DataFrame, gt_column="text_category"):
    """
    main function. maps predicted clusters to ground truth clusters. gives an score for the match.
    expected columns: 'class_gt' with numeric labels for ground truth clusters or 'text_category' with text
                      'class_prd' with numeric labels for predicted clusters or 'predicted_topic' with text
    Returns:
    -------
    df: pd.DataFrame with mapped clusters in 'mapped_cluster' and corresponding f-measure in 'max_fmeasure' column
    fmeasure_aggregate: float
                        average f-measure as over-all performance metric
    true_dct: pd.DataFrame with clusters which match exactly
    """
    
    if gt_column!="text_category" and gt_column!='class_gt':
        df.rename(columns={gt_column: "text_category"}, inplace=True)

    if 'class_gt' not in df.columns:
        df['summary_md'] = df['text_category'].apply(preprocess)
        le_gt = LabelEncoder()
        df['class_gt'] = le_gt.fit_transform(df['summary_md'])
    if 'class_prd' not in df.columns:
        df['predicted_topic_md'] = df['predicted_topic'].apply(preprocess)
        le_prd = LabelEncoder()
        df['class_prd'] = le_prd.fit_transform(df['predicted_topic_md'])

    df = join_count(df, 'class_gt', 'gt_class_count')
    df = join_count(df, 'class_prd', 'prd_class_count')
    count_agg = eval_fmeasure(df)
    true_match = count_agg[count_agg['fmeasure']==1]
    unmatched = count_agg[count_agg['fmeasure']!=1]
    rough_match = map_rough_clusters(unmatched)
    df['max_fmeasure'] = 0
    df['mapped_cluster'] = -1
    if len(true_match)>0:
        true_dct = true_match.reset_index().groupby('class_gt').apply(lambda x: (x['class_prd'].values[0], x['fmeasure'].values[0]))
    else:
        true_dct = pd.DataFrame()
        true_ind = []
    true_ind = list(true_dct.index)
    df['mapped_cluster'] = df['class_gt'].apply(lambda x: rough_match[x][0] if x in rough_match.keys() else -1)
    df['max_fmeasure'] = df['class_gt'].apply(lambda x: rough_match[x][1] if x in rough_match.keys() else 0)
    df['mapped_cluster'] = df.apply(lambda x: true_dct.loc[x['class_gt']][0] if x['class_gt'] in true_ind else x['mapped_cluster'], axis=1)
    df['max_fmeasure'] = df.apply(lambda x: true_dct.loc[x['class_gt']][1] if x['class_gt'] in true_ind else x['max_fmeasure'], axis=1)
    fmeasure_aggregate = sum(df['max_fmeasure'])/(len(df)+max(0, len(df['class_prd'].unique())-len(df['class_gt'].unique())))
    return df, fmeasure_aggregate, true_dct


if __name__ == "__main__":
    caller = ArgumentParser()
    caller.add_argument("input", help="path to input csv")
    caller.add_argument("output", help="path to save output csv")
    
    z = caller.parse_args()
    inp = pd.read_csv(z.input)
    out, fm, _ = cluster_measure(inp)
    print(fm)
    out.to_csv(z.output)
