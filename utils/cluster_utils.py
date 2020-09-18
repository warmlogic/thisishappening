import logging

import numpy as np
from sklearn.cluster import DBSCAN

from utils.data_base import Tiles

logger = logging.getLogger("happeninglogger")


def cluster_activity(session, activity, min_samples: int, kms: float = 0.1, min_n_clusters: int = 1, max_tries: int = 8):
    kms_per_radian = 6371.0088
    eps = kms / kms_per_radian
    eps_step = eps / 2.0

    # haversine metric requires radians
    X = np.radians([[x.longitude, x.latitude] for x in activity])

    clusters = {}
    unique_labels = []
    n_tries = 0
    while len(unique_labels) < min_n_clusters:
        n_tries += 1

        if n_tries > max_tries:
            logger.info(f'Tried maximum number of times ({max_tries}), not continuing')
            break

        logger.info(f'Running clustering attempt {n_tries}')
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(X)

        # label -1 means not assigned to a cluster
        unique_labels = [x for x in set(db.labels_) if x != -1]
        logger.info(f'Found {len(unique_labels)} clusters')

        if len(unique_labels) < min_n_clusters:
            logger.info(f'Increasing epsilon from {eps} to {eps + eps_step}')
            eps += eps_step

    if unique_labels:
        for k in unique_labels:
            cluster_mask = (db.labels_ == k)
            cluster_tweets = [x for i, x in enumerate(activity) if cluster_mask[i]]

            # Compute the average tweet location
            lons = [x.longitude for x in cluster_tweets]
            longitude = sum(lons) / len(lons)
            lats = [x.latitude for x in cluster_tweets]
            latitude = sum(lats) / len(lats)

            # Find the tile that contains this location, for naming
            tiles = Tiles.find_id_by_coords(session, longitude, latitude)
            tile_id = tiles[0].id

            clusters[k] = {
                'event_tweets': cluster_tweets,
                'tile_id': tile_id,
            }

    return clusters
