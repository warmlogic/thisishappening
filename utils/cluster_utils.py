import logging

import numpy as np
from sklearn.cluster import DBSCAN

from utils.data_base import Tiles

logger = logging.getLogger("happeninglogger")


def cluster_activity(session, activity, min_samples: int, eps: float = 0.075, metric: str = 'euclidean'):

    X = np.array([[x.longitude, x.latitude] for x in activity])
    clusters = {}

    n_clusters = 0
    min_n_clusters = 1
    eps_step = eps / 2
    n_tries = 0
    max_tries = 8

    while n_clusters < min_n_clusters:
        n_tries += 1

        if n_tries > max_tries:
            logger.info(f'Tried maximum number of times ({max_tries}), not continuing')
            break

        logger.info(f'Running clustering attempt {n_tries}')
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)

        labels = db.labels_
        unique_labels = [x for x in set(labels) if x != -1]
        n_clusters = len(unique_labels)
        logger.info(f'Found {n_clusters} clusters')

        if n_clusters < min_n_clusters:
            logger.info(f'Increasing epsilon from {eps} to {eps + eps_step}')
            eps += eps_step

    if n_clusters:
        for k in n_clusters:
            cluster_mask = (labels == k)
            cluster_tweets = activity[cluster_mask]

            # Compute the average tweet location
            lons = [x.longitude for x in cluster_tweets]
            longitude = sum(lons) / len(lons)
            lats = [x.latitude for x in cluster_tweets]
            latitude = sum(lats) / len(lats)

            # Find the tile that contains this location, for naming
            tile_id = Tiles.find_id_by_coords(session, longitude, latitude)

            clusters[k] = {
                'event_tweets': cluster_tweets,
                'tile_id': tile_id,
            }

    return clusters
