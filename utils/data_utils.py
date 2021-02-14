import itertools
import logging
from operator import itemgetter
from time import sleep
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger("happeninglogger")


def n_wise(iterable: List, n: int) -> zip(Tuple):
    """n_wise - Given an iterable, create a generator of successive groups of size n

    list(n_wise([1, 2, 3, 4, 5], 3)) -> [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Parameters
    ----------
    iterable : List (or any iterable)
        Items to include in groups
    n : int
        Group size

    Returns
    -------
    zip generator of tuples
        Items in groups
    """
    return zip(*(itertools.islice(iterable, i, None) for i in range(n)))


def inbounds(longitude, latitude, bounding_box: List[float]):
    lon = (longitude >= bounding_box[0]) and (longitude <= bounding_box[2])
    lat = (latitude >= bounding_box[1]) and (latitude <= bounding_box[3])
    return (lon and lat)


def reverse_geocode(twitter, longitude: float, latitude: float) -> Dict:
    # Reverse geocode the tile's center latitude and longitude value to store tile names
    unsuccessful_tries = 0
    try_threshold = 10

    rev_geo = {}
    while unsuccessful_tries < try_threshold:
        rev_geo = twitter.reverse_geocode(lat=latitude, long=longitude, granularity='neighborhood')
        if 'result' in rev_geo:
            unsuccessful_tries = try_threshold
        else:
            unsuccessful_tries += 1
            logger.info('Sleeping for 10 seconds due to failed reverse geocode')
            sleep(10)

    return rev_geo


def get_coords_min_max(bounding_box: List[float]):
    # longitude
    xmin, xmax = bounding_box[0], bounding_box[2]
    # latitude
    ymin, ymax = bounding_box[1], bounding_box[3]
    return xmin, xmax, ymin, ymax


def get_grid_coords(bounding_box: List[float]):
    xmin, xmax, ymin, ymax = get_coords_min_max(bounding_box)
    x_flat = np.r_[xmin:xmax:128j]
    y_flat = np.r_[ymin:ymax:128j][::-1]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    return grid_coords, x_flat, y_flat


def compute_weight(x: int, weight_factor: float = None) -> float:
    weight_factor = 1.0 if weight_factor is None else weight_factor
    return 1 / np.exp(x * weight_factor)


def set_activity_weight(activity, weighted: bool = None, weight_factor: float = None) -> list:
    weighted = True if weighted is None else weighted
    # Compute tweet weight within a user
    activity_sorted = sorted([x.__dict__ for x in activity], key=itemgetter("user_id_str"))
    activity_grouped = {}
    for user_id, tweets in itertools.groupby(activity_sorted, key=lambda x: x["user_id_str"]):
        # Sort user tweets so first tweet has highest weight
        activity_grouped[user_id] = sorted(tweets, key=itemgetter("created_at"))
        for i, tweet in enumerate(activity_grouped[user_id]):
            tweet["weight"] = compute_weight(i, weight_factor) if weighted else 1.0

    # Get a flat list of tweets
    activity_weighted = [tweet for tweets in list(activity_grouped.values()) for tweet in tweets]

    return activity_weighted


def get_kde(grid_coords, activity, bw_method: float = None, weighted: bool = None, weight_factor: float = None):
    bw_method = 0.3 if bw_method is None else bw_method

    activity_weighted = set_activity_weight(activity, weighted=weighted, weight_factor=weight_factor)
    sample_weight = [x["weight"] for x in activity_weighted]

    lon_lat = np.array([[x["longitude"], x["latitude"]] for x in activity_weighted])

    kernel = stats.gaussian_kde(lon_lat.T, bw_method=bw_method, weights=sample_weight)

    z = kernel(grid_coords.T)
    z = z.reshape(128, 128)

    return z, kernel, activity_weighted


def compare_activity_kde(
    grid_coords,
    activity_prev, activity_curr,
    bw_method: float = None,
    weighted: bool = None, weight_factor: float = None,
):
    z_prev, _, activity_prev_weighted = get_kde(grid_coords, activity_prev, bw_method=bw_method, weighted=weighted, weight_factor=weight_factor)
    z_curr, _, activity_curr_weighted = get_kde(grid_coords, activity_curr, bw_method=bw_method, weighted=weighted, weight_factor=weight_factor)
    z_diff = z_curr - z_prev

    return z_diff, activity_prev_weighted, activity_curr_weighted
