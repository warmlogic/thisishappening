import logging

import numpy as np
from sklearn.cluster import DBSCAN

from utils.data_base import Tiles, RecentTweets

logger = logging.getLogger("happeninglogger")

KMS_PER_RADIAN = 6371.0088


def cluster_activity(session, activity, min_samples: int, kms: float = 0.1, min_n_clusters: int = 1, max_tries: int = 5, sample_weight=None):
    if not activity:
        return {}

    eps = kms / KMS_PER_RADIAN
    eps_step = eps / 2.0

    # haversine metric requires radians
    try:
        X = np.radians([[x.longitude, x.latitude] for x in activity])
    except AttributeError:
        X = np.radians([[x['longitude'], x['latitude']] for x in activity])
    except TypeError:
        logger.exception(f"Unsupported activity dtype: {type(activity[0])}")

    clusters = {}
    unique_labels = []
    n_tries = 0
    while len(unique_labels) < min_n_clusters:
        n_tries += 1

        if n_tries > max_tries:
            logger.info(f'Tried maximum number of times ({max_tries}), not continuing')
            break

        logger.info(f'Running clustering attempt {n_tries}')
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
        db.fit(X, sample_weight=sample_weight)

        # label -1 means not assigned to a cluster
        unique_labels = [x for x in set(db.labels_) if x != -1]
        logger.info(f'Found {len(unique_labels)} clusters')

        if len(unique_labels) < min_n_clusters:
            logger.info(f'Increasing epsilon from {eps} to {eps + eps_step}')
            eps += eps_step

    if len(unique_labels) > 0:
        for k in unique_labels:
            cluster_mask = (db.labels_ == k)
            cluster_tweets = [x for i, x in enumerate(activity) if cluster_mask[i]]

            # Compute the average tweet location
            try:
                lons = [x.longitude for x in cluster_tweets]
                lats = [x.latitude for x in cluster_tweets]
            except AttributeError:
                lons = [x['longitude'] for x in cluster_tweets]
                lats = [x['latitude'] for x in cluster_tweets]
            except TypeError:
                logger.exception(f"Unsupported cluster tweet dtype: {type(cluster_tweets[0])}")

            longitude = sum(lons) / len(lons)
            latitude = sum(lats) / len(lats)

            # Find the tile that contains this location, for naming
            tiles = Tiles.find_id_by_coords(session, longitude, latitude)
            tile_id = tiles[0].id

            clusters[k] = {
                'event_tweets': cluster_tweets,
                'tile_id': tile_id,
            }

    return clusters
