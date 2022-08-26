from typing import List
from collections import namedtuple

from src.utils import validate_format
from src.utils import cache
from src.reader import Reader
import pandas as pd


VALID_DATA_FORMATS = ["UIR", "UIRT"]

MovieLens = namedtuple("MovieLens", ["url", "unzip", "path", "sep", "skip"])
ML_DATASETS = {
    "100K": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        False,
        "ml-100k/u.data",
        "\t",
        0,
    ),
    "1M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        True,
        "ml-1m/ratings.dat",
        "::",
        0,
    ),
    "10M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
        True,
        "ml-10M100K/ratings.dat",
        "::",
        0,
    ),
    "20M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        True,
        "ml-20m/ratings.csv",
        ",",
        1,
    ),
}


def load_feedback(fmt="UIR", variant="100K", reader=None):
    """Load the user-item ratings of one of the MovieLens datasets
    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned, one of ['UIR', 'UIRT'].
    variant: str, optional, default: '100K'
        Specifies which MovieLens dataset to load, one of ['100K', '1M', '10M', '20M'].
    reader: `obj:src.reader.Reader`, optional, default: None
        Reader object used to read the data.
    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.
    """

    fmt = validate_format(fmt, VALID_DATA_FORMATS)

    ml = ML_DATASETS.get(variant.upper(), None)
    if ml is None:
        raise ValueError("variant must be one of {}.".format(ML_DATASETS.keys()))

    fpath = cache(url=ml.url, unzip=ml.unzip, relative_path=ml.path)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt, sep=ml.sep, skip_lines=ml.skip)