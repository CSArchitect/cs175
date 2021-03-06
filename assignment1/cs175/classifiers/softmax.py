import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      scores = X[i, :].dot(W)
      scores -= np.max(scores)
      correct_scores = scores[y[i]]
      score_sum = np.sum(np.exp(scores))
      h = np.exp(correct_scores) / score_sum
      loss += -np.log(h)
      for j in xrange(num_classes):
          if j == y[i]:
              dW[:, y[i]] += (np.exp(scores[j]) / score_sum - 1) * X[i, :]
          else:
              dW[:, j] += (np.exp(scores[j]) / score_sum) * X[i, :]
                
                
  loss /= num_train + ( reg * np.sum(W * W))
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  scores -= np.max(scores)
  e_scores = np.exp(scores)
  score_sum = np.sum(e_scores, axis = 1)
  score_sum = np.atleast_2d(score_sum).T
  h_x = (1/(score_sum)) * e_scores
  
  onehot = np.zeros((X.shape[0], W.shape[1]))
  index = np.arange(X.shape[0])
  onehot[index, y] = 1

  loss = -1*onehot * np.log(h_x)
  loss = np.mean(np.sum(loss, axis = 1))
    
    
  #gradient
  dW = ((1/X.shape[0]) * np.dot(X.T,(h_x - onehot))) + 2* reg * W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

