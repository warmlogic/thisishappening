import logging

import numpy as np
from geopy.distance import EARTH_RADIUS as KMS_PER_RADIAN
from sklearn.cluster import DBSCAN

from .tweet_utils import get_coords

logger = logging.getLogger("happeninglogger")


def cluster_activity(
    activity,
    min_samples: int,
    km_start: float = 0.05,
    km_stop: float = 0.25,
    km_step: int = 5,
    min_n_clusters: int = 1,
    sample_weight=None,
):
    if len(activity) == 0:
        return {}

    kms = np.linspace(km_start, km_stop, km_step)
    _eps = [(km / KMS_PER_RADIAN) for km in kms]

    # haversine metric requires radians
    lons, lats = get_coords(activity)
    X = np.radians([[lon, lat] for lon, lat in zip(lons, lats)])

    unique_labels = []
    for km, eps in zip(kms, _eps):
        db = DBSCAN(
            eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
        )
        db.fit(X, sample_weight=sample_weight)

        # label -1 means not assigned to a cluster
        unique_labels = [x for x in set(db.labels_) if x != -1]

        if len(unique_labels) >= min_n_clusters:
            break

    logger.info(
        f"Clustered to max neighbor distance {km:.3f} km,"
        + f" found {len(unique_labels)} clusters"
    )

    clusters = {}
    for k in unique_labels:
        cluster_mask = db.labels_ == k
        cluster_tweets = [x for i, x in enumerate(activity) if cluster_mask[i]]
        clusters[k] = {
            "event_tweets": cluster_tweets,
        }

    return clusters
