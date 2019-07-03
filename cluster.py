from sklearn.cluster import DBSCAN
from itertools import groupby

def get_clusters(pts, key=lambda x: x, max_clusters=None, max_distance=14):
    """Run DBSCAN on the `pts`, applying `key` first if necessary,
    post-process the results into a list of lists, and return it,
    taking only the largest `max_clusters` clusters.
    """
    if pts:
        kpts = [key(pt) for pt in pts]

        clustering = DBSCAN(eps=max_distance, min_samples=1).fit(kpts)

        # Post-processing.
        labeled_pts = list(zip(kpts, clustering.labels_))
        labeled_pts = sorted(labeled_pts, key=lambda p: p[1])

        clusters = [list(g) for l, g in groupby(labeled_pts, key=lambda p: p[1])]
        clusters = [[p[0] for p in clust] for clust in clusters]
        clusters = list(sorted(clusters, key=len, reverse=True))

        return clusters[:max_clusters]
    return []
