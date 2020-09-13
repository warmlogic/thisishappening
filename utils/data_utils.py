from itertools import islice
from typing import List, Tuple


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
    return zip(*(islice(iterable, i, None) for i in range(n)))
