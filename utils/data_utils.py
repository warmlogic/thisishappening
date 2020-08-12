from itertools import islice


def n_wise(iterable, n):
    return zip(*(islice(iterable, i, None) for i in range(n)))
