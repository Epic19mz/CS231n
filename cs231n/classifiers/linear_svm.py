import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:            
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                dW[:,y[i]]-=X[i,:].T
                dW[:,j]+=X[i,:].T
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss/= num_train
    dW/=num_train
    dW+=reg*2*W
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    list(range(y.shape[0]))
    margin=np.asarray([-X.dot(W)[list(range(y.shape[0])),list(y)]]).T+X.dot(W)+1
    loss=np.sum((margin>0)*margin)/X.shape[0]+reg*np.sum(W*W)-1
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    temp=np.ones(margin.shape)*(margin>0)
    temp[list(range(temp.shape[0])),list(y)]=0
    dW=X.T.dot(temp)   
    temp1=np.zeros((X.shape[0],len(set(y))))
    temp1[list(range(temp1.shape[0])),list(y)]=1
    print(((np.sum(temp,axis=1,keepdims=True)*X).T).shape)
    print(temp1.shape)
    dW-=((np.sum(temp,axis=1,keepdims=True)*X).T).dot(temp1)
    dW=dW/X.shape[0]+2*reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW
