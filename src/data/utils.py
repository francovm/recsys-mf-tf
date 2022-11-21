import os
import shutil
import zipfile
import tarfile
from urllib import request
import numpy as np
import pandas as pd
from IPython import display

from tqdm.auto import tqdm

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def _urlretrieve(url, fpath):
    """Retrieve data from given url
    Parameters
    ----------
    url: str
        The url to the data.
    fpath: str
        The path to file where data is stored.
    """
    opener = request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]

    with tqdm(unit="B", unit_scale=True) as progress:

        def report(chunk, chunksize, total):
            progress.total = total
            progress.update(chunksize)

        request.install_opener(opener)
        request.urlretrieve(url, fpath, reporthook=report)


def _extract_archive(file_path, extract_path="."):
    """Extracts an archive.
    """
    for archive_type in ["zip", "tar"]:
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        elif archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(extract_path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(extract_path):
                        if os.path.isfile(extract_path):
                            os.remove(extract_path)
                        else:
                            shutil.rmtree(extract_path)
                    raise


def validate_format(input_format, valid_formats):
    """Check the input format is in list of valid formats
    :raise ValueError if not supported
    """
    if not input_format in valid_formats:
        raise ValueError('{} data format is not in valid formats ({})'.format(
            input_format, valid_formats))

    return input_format


def get_cache_path(relative_path, cache_dir=None):
    """Return the absolute path to the cached data file
    """
    if cache_dir is None and os.access(os.path.expanduser("~"), os.W_OK):
        cache_dir = os.path.join(os.path.expanduser("~"), ".recsys-mf")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    if not os.access(cache_dir, os.W_OK):
        cache_dir = os.path.join("/tmp", ".recsys-mf")
    cache_path = os.path.join(cache_dir, relative_path)

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    return cache_path, cache_dir


def cache(url, unzip=False, relative_path=None, cache_dir=None):
    """Download the data and cache to file
    Parameters
    ----------
    url: str
        The url to the data.
    unzip: bool, optional, default: False
        Whether the data is a zip file and going to be unzipped after the download.
    relative_path: str
        Relative path to the data file after finishing the download.
        If unzip=True, relative_path is the path to unzipped file.
    cache_dir: str, optional, default: None
        The path to cache folder. If `None`, either ~/.recsys-mf or /tmp/.recsys-mf will be used.
    """
    if relative_path is None:
        relative_path = url.split("/")[-1]
    cache_path, cache_dir = get_cache_path(relative_path, cache_dir)
    if os.path.exists(cache_path):
        return cache_path

    print("Data from", url)
    print("will be cached into", cache_path)

    if unzip:
        tmp_path = os.path.join(cache_dir, "file.tmp")
        _urlretrieve(url, tmp_path)
        print("Unzipping ...")
        _extract_archive(tmp_path, cache_dir)
        os.remove(tmp_path)
    else:
        _urlretrieve(url, cache_path)

    print("File cached!")
    return


def build_rating_sparse_tensor(ratings_df,Tensor_shape):
  """
  Args:
    ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
  Returns:
    a tf.SparseTensor representing the ratings matrix.
  """
  ratings_df["item_id"] = ratings_df["item_id"].astype(int)
  ratings_df["user_id"] = ratings_df["user_id"].astype(int)
  ratings_df["rating"] = ratings_df["rating"].astype(float)

  ratings_df["item_id"] = ratings_df["item_id"].apply(lambda x: str(x-1))
  ratings_df["user_id"] = ratings_df["user_id"].apply(lambda x: str(x-1))
  ratings_df["rating"] = ratings_df["rating"].apply(lambda x: float(x)) 

  indices = ratings_df[['user_id', 'item_id']].values
  values = ratings_df['rating'].values
  return tf.SparseTensor(
    indices=indices,
    values=values,
    dense_shape=Tensor_shape)
    

def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
    df: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
    train: dataframe for training
    test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test

# def compute_scores(query_embedding, item_embeddings, cosine=True):
#   """Computes the scores of the candidates given a query.
#   Args:
#     query_embedding: a vector of shape [k], representing the query embedding.
#     item_embeddings: a matrix of shape [N, k], such that row i is the embedding
#       of item i.
#     measure: a string specifying the similarity measure to be used. Can be
#       either DOT or COSINE.
#   Returns:
#     scores: a vector of shape [N], such that scores[i] is the score of item i.
#   """
#   u = query_embedding
#   V = item_embeddings
#   if cosine is True:
#     V = V / np.linalg.norm(V, axis=1, keepdims=True)
#     u = u / np.linalg.norm(u)
#   scores = u.dot(V.T)
#   return scores
    
# def movie_neighbors(model, movies, title_substring,cosine=True, k=6):
#     """
#     Generate movie recommendations.
#     """
#     # Search for movie ids that match the given substring.
#     ids =  movies[movies['title'].str.contains(title_substring)].index.values
#     titles = movies.iloc[ids]['title'].values
#     if len(titles) == 0:
#      raise ValueError("Found no movies with title %s" % title_substring)
#     print("Nearest neighbors of : %s." % titles[0])
#     if len(titles) > 1:
#      print("[Found more than one matching movie. Other candidates: {}]".format(
#          ", ".join(titles[1:])))
#     item_id = ids[0]
#     scores = compute_scores(
#        model.embeddings["movie_id"][item_id], model.embeddings["movie_id"],
#        cosine)
#     # score_key = measure + ' score'
#     df = pd.DataFrame({
#        'score': list(scores),
#        'titles': movies['title'],
#        'genres': movies['all_genres']
#     })
#     display.display(df.sort_values(['score'], ascending=False).head(k))

