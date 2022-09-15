from src.model.recommender import Recommender
from src.data import build_rating_sparse_tensor, split_dataframe


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class MFmodel:
    """
    """
    def __init__(self, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
        init_stddev=0.1):

        self.embedding_dim = embedding_dim
        self.regularization_coeff = regularization_coeff
        self.gravity_coeff = gravity_coeff
        self.init_stddev = init_stddev

    def _sparse_mean_square_error(self,sparse_ratings, user_embeddings, movie_embeddings):
        """ A TensorFlow function that takes a sparse rating matrix  A  and the two embedding matrices  U,V  and returns the mean squared error  MSE(A,UVT) .
        Args:
        sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
        user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
          dimension, such that U_i is the embedding of user i.
        movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
          dimension, such that V_j is the embedding of movie j.
        Returns:
        A scalar Tensor representing the MSE between the true ratings and the
          model's predictions.
        """
        predictions = tf.reduce_sum(
          tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
          tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]),
          axis=1)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss

    def _gravity(self,U, V):
      """Creates a gravity loss given two embedding matrices."""
      return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
          tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

    def fit(self, ratings):
      """
      Args:
        ratings: the DataFrame of movie ratings.
        embedding_dim: The dimension of the embedding space.
        regularization_coeff: The regularization coefficient lambda.
        gravity_coeff: The gravity regularization coefficient lambda_g.
      Returns:
        A MF Model object that uses a regularized loss.
      """
      # Define origianl shape before splitting data
      input_shape = [ratings.user_id.nunique()+1, ratings.item_id.nunique()+1]
      # Split the ratings DataFrame into train and test.
      train_ratings, test_ratings = split_dataframe(ratings)
      # SparseTensor representation of the train and test datasets.
      A_train = build_rating_sparse_tensor(train_ratings,input_shape)
      A_test = build_rating_sparse_tensor(test_ratings,input_shape)
      U = tf.Variable(tf.random_normal(
          [A_train.dense_shape[0], self.embedding_dim], stddev=self.init_stddev))
      V = tf.Variable(tf.random_normal(
          [A_train.dense_shape[1], self.embedding_dim], stddev=self.init_stddev))

      error_train = self._sparse_mean_square_error(A_train, U, V)
      error_test = self._sparse_mean_square_error(A_test, U, V)
      gravity_loss = self.gravity_coeff * self._gravity(U, V)
      regularization_loss = self.regularization_coeff * (
          tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)
      total_loss = error_train + regularization_loss + gravity_loss
      losses = {
          'train_error_observed': error_train,
          'test_error_observed': error_test,
      }
      loss_components = {
          'observed_loss': error_train,
          'regularization_loss': regularization_loss,
          'gravity_loss': gravity_loss,
      }
      embeddings = {"user_id": U, "movie_id": V}

      return Recommender(embeddings, total_loss, [losses, loss_components])