#!/usr/bin/python

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from itertools import groupby

df = pd.read_csv('/home/sasha/work/meleedb-segment/ts.csv')

df_ = df[['time', 'median']]

for scale in np.arange(1, 5, 0.05):
    df_['time'] = (df['time'] / max(df['time'])) * scale

    clust = DBSCAN(eps=.05, min_samples=7)

    df_['clust'] = clust.fit_predict(df_)
    outliers = df_['clust'].value_counts()[-1]
    df_['clust'] = df_['clust'].replace(-1, None).fillna(method='bfill')

    groups = groupby([r for n, r in df_.iterrows()], key=lambda r: r['clust'])
    groups = [(k, list(g)) for k, g in groups]

    bad_groups = []
    for idx, group in enumerate(groups[1:]):
        num, g = group
        if num < groups[idx][0]:
            bad_groups.append(list(g))

    score = (outliers + len(groups))**2 * (1 + sum(len(g) for g in bad_groups))

    max_width = 0
    for k, g in groups:
        medians = [r['median'] for r in g]
        width = max(medians) - min(medians)
        max_width = max(width, max_width)

    score **= 1 - max_width

    print(scale, score)
