# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss_mae(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.sum(np.abs(y-tx@w))/y.shape[0]

def compute_loss_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e=y-tx.dot(w)
    return 1/(2*y.shape[0]) * e.dot(e)