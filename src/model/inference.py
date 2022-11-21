import numpy as np
import pandas as pd
from IPython import display

def compute_scores(query_embedding, item_embeddings, cosine=True):
  """Computes the scores of the candidates given a query.
  Args:
    query_embedding: a vector of shape [k], representing the query embedding.
    item_embeddings: a matrix of shape [N, k], such that row i is the embedding
      of item i.
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
  Returns:
    scores: a vector of shape [N], such that scores[i] is the score of item i.
  """
  u = query_embedding
  V = item_embeddings
  if cosine is True:
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    u = u / np.linalg.norm(u)
  scores = u.dot(V.T)
  return scores
    
def movie_neighbors(model, movies, title_substring,cosine=True, k=6):
    """
    Generate movie recommendations.
    """
    # Search for movie ids that match the given substring.
    ids =  movies[movies['title'].str.contains(title_substring)].index.values
    titles = movies.iloc[ids]['title'].values
    if len(titles) == 0:
     raise ValueError("Found no movies with title %s" % title_substring)
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
     print("[Found more than one matching movie. Other candidates: {}]".format(
         ", ".join(titles[1:])))
    item_id = ids[0]
    scores = compute_scores(
       model.embeddings["movie_id"][item_id], model.embeddings["movie_id"],
       cosine)
    # score_key = measure + ' score'
    df = pd.DataFrame({
       'score': list(scores),
       'titles': movies['title'],
       'genres': movies['all_genres']
    })
    display.display(df.sort_values(['score'], ascending=False).head(k))