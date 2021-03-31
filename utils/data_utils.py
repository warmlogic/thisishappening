import itertools
import logging
from operator import itemgetter
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


def inbounds(longitude: float, latitude: float, bounding_box: List[float]) -> bool:
    lon = (longitude >= bounding_box[0]) and (longitude <= bounding_box[2])
    lat = (latitude >= bounding_box[1]) and (latitude <= bounding_box[3])
    return (lon and lat)


def get_coords_min_max(bounding_box: List[float]):
    # longitude: xmin=west_lon, xmax=east_lon
    xmin, xmax = bounding_box[0], bounding_box[2]
    # latitude: ymin=south_lat, ymax=north_lat
    ymin, ymax = bounding_box[1], bounding_box[3]
    return xmin, xmax, ymin, ymax


def get_grid_coords(bounding_box: List[float], grid_resolution: int):
    xmin, xmax, ymin, ymax = get_coords_min_max(bounding_box)
    x_flat = np.linspace(xmin, xmax, grid_resolution)
    # y is reversed
    y_flat = np.linspace(ymax, ymin, grid_resolution)
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    return grid_coords, x_flat, y_flat


def compute_weight(weight: float, x: int, weight_factor: float = None) -> float:
    weight_factor = 1.0 if weight_factor is None else weight_factor
    return weight / np.exp(x * weight_factor)


def set_activity_weight(
    activity,
    weighted: bool = None,
    weight_factor: float = None,
    reduce_weight_lon_lat: List[Tuple[float, float]] = None,
    weight_factor_lon_lat: float = None,
    weight_factor_no_coords: float = None,
) -> List[Dict]:
    weighted = True if weighted is None else weighted
    reduce_weight_lon_lat = [] if reduce_weight_lon_lat is None else reduce_weight_lon_lat
    weight_factor_lon_lat = 2.0 if weight_factor_lon_lat is None else weight_factor_lon_lat
    weight_factor_no_coords = 4.0 if weight_factor_no_coords is None else weight_factor_no_coords

    # Create a list of dictionaries and remove the sqlalchemy instance state key
    activity_dict = [{k: v for k, v in x.__dict__.items() if k != "_sa_instance_state"} for x in activity]

    # Give every tweet a weight
    for tweet in activity_dict:
        tweet["weight"] = 1.0

    # Reduce weight if tweet has specific coordinates
    if weighted and reduce_weight_lon_lat:
        for tweet in activity_dict:
            if (tweet["longitude"], tweet["latitude"]) in reduce_weight_lon_lat:
                tweet["weight"] = compute_weight(tweet["weight"], 1, weight_factor_lon_lat)

    # Reduce weight if tweet did not have specific coordinates
    if weighted and (weight_factor_no_coords is not None):
        for tweet in activity_dict:
            if not tweet["has_coords"]:
                tweet["weight"] = compute_weight(tweet["weight"], 1, weight_factor_no_coords)

    # Compute tweet weight within a user
    activity_sorted = sorted(activity_dict, key=itemgetter("user_id_str"))
    activity_grouped = {}
    for user_id, tweets in itertools.groupby(activity_sorted, key=lambda x: x["user_id_str"]):
        # Sort user tweets so first tweet has highest weight
        activity_grouped[user_id] = sorted(tweets, key=itemgetter("created_at"))
        if weighted:
            for i, tweet in enumerate(activity_grouped[user_id]):
                tweet["weight"] = compute_weight(tweet["weight"], i, weight_factor)

    # Get a flat list of tweets
    activity_weighted = [tweet for tweets in list(activity_grouped.values()) for tweet in tweets]

    return activity_weighted


def get_kde(
    grid_coords,
    activity,
    bw_method=None,
    weighted: bool = None,
    weight_factor: float = None,
    reduce_weight_lon_lat: List[Tuple[float, float]] = None,
    weight_factor_lon_lat: float = None,
    weight_factor_no_coords: float = None,
):
    bw_method = 0.3 if bw_method is None else bw_method
    gc_shape = int(np.sqrt(grid_coords.shape[0]))

    activity_weighted = set_activity_weight(
        activity,
        weighted=weighted,
        weight_factor=weight_factor,
        reduce_weight_lon_lat=reduce_weight_lon_lat,
        weight_factor_lon_lat=weight_factor_lon_lat,
        weight_factor_no_coords=weight_factor_no_coords,
    )
    sample_weight = np.array([x["weight"] for x in activity_weighted])

    lon_lat = np.array([[x["longitude"], x["latitude"]] for x in activity_weighted])

    try:
        kernel = stats.gaussian_kde(lon_lat.T, bw_method=bw_method, weights=sample_weight)
    except np.linalg.LinAlgError as e:
        logger.info(f"Could not get kernel density estimate, {e}")
        kernel = None

    if kernel is not None:
        try:
            z = kernel(grid_coords.T)
            z = z.reshape(gc_shape, gc_shape)
        except np.linalg.LinAlgError as e:
            logger.info(f"Could not use kernel, {e}")
            z = np.zeros([gc_shape, gc_shape])
    else:
        z = np.zeros([gc_shape, gc_shape])

    return z, kernel, activity_weighted


def compare_activity_kde(
    grid_coords,
    activity_prev,
    activity_curr,
    bw_method: float = None,
    weighted: bool = None,
    weight_factor: float = None,
    reduce_weight_lon_lat: List[Tuple[float, float]] = None,
    weight_factor_lon_lat: float = None,
    weight_factor_no_coords: float = None,
):
    z_prev, _, activity_prev_weighted = get_kde(
        grid_coords,
        activity_prev,
        bw_method=bw_method,
        weighted=weighted,
        weight_factor=weight_factor,
        reduce_weight_lon_lat=reduce_weight_lon_lat,
        weight_factor_lon_lat=weight_factor_lon_lat,
        weight_factor_no_coords=weight_factor_no_coords,
    )
    z_curr, _, activity_curr_weighted = get_kde(
        grid_coords,
        activity_curr,
        bw_method=bw_method,
        weighted=weighted,
        weight_factor=weight_factor,
        reduce_weight_lon_lat=reduce_weight_lon_lat,
        weight_factor_lon_lat=weight_factor_lon_lat,
        weight_factor_no_coords=weight_factor_no_coords,
    )
    z_diff = z_curr - z_prev

    return z_diff, activity_prev_weighted, activity_curr_weighted
