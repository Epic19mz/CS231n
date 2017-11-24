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
  scores=X.dot(W)
  for i in range(X.shape[0]):
        loss+=-np.log(np.exp(scores[i,y[i]])/sum(np.exp(scores[i,])))
        dW[:,y[i]]+=-X[i,]+(np.exp(scores[i,y[i]]))/np.sum(np.exp(scores[i,]))*X[i,]
        temp=(np.asarray([X[i,]]).T).dot(np.asarray([np.exp(scores[i,])]))/np.sum(np.exp(scores[i,]))
        dW+=temp
        dW[:,y[i]]-=temp[:,y[i]]
  loss/=X.shape[0]
  dW/=X.shape[0]
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
  scores=X.dot(W)
  loss=np.mean(-np.log(np.exp(scores)[list(range(scores.shape[0])),list(y)]/np.sum(np.exp(scores),axis=1)))
  dW=X.T.dot(np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True))
  temp=np.zeros((X.shape[0],len(set(y))))
  temp[list(range(temp.shape[0])),list(y)]=1
  dW+=-(X.T).dot(temp)
  dW/=X.shape[0]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

