# code: -*- utf-8 -*-
# hacked by hwang from sci-kit learn
# original code can be found at:
# https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/classification.py
import math

import numpy as np

from sklearn.utils import check_array
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.utils import check_consistent_length

EPS_ = 1e-15

def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def log_loss_0(w):
    '''
    a simple version of cross entropy loss
    '''
    X = np.array([[0,2]])
    y_true = 1
    y_pred= np.dot(w, np.transpose(X))
    # Clipping
    y_pred = np.clip(y_pred, EPS_, 1 - EPS_)
    loss = -(y_true*math.log(y_pred)+(1-y_pred)*math.log(1-y_pred))
    return loss

def log_loss_1(w):
    '''
    a simple version of cross entropy loss
    '''
    X = np.array([[2,0]])
    y_true = -1
    y_pred= np.dot(w, np.transpose(X))
    # Clipping
    y_pred = np.clip(y_pred, EPS_, 1 - EPS_)
    loss = -(y_true*math.log(y_pred)+(1-y_pred)*math.log(1-y_pred))
    return loss

'''
def log_loss_0(w):
    """Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of the true labels given a probabilistic classifier's
    predictions. The log loss is only defined for two or more labels.
    For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is
        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    Read more in the :ref:`User Guide <log_loss>`.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.
    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).
    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    labels : array-like, optional (default=None)
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.
        .. versionadded:: 0.18
    Returns
    -------
    loss : float
    Examples
    --------
    >>> log_loss(["spam", "ham", "ham", "spam"],  # doctest: +ELLIPSIS
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...
    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    # do this hack for hessian computation
    eps=1e-15
    normalize=True
    sample_weight=None
    labels=None
    # define labels here:
    X = np.array([[0,2], [2, 0]])
    y_true=np.array([1,0])
#    X = np.array([[0,2]])
#    y_true=np.array([1])
#    y_true=1
    y_pred= np.dot(np.transpose(w), X)

    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please '
                             'provide the true labels explicitly through the '
                             'labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two '
                             'labels for log_loss, '
                             'got {0}.'.format(lb.classes_))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError("y_true and y_pred contain different number of "
                             "classes {0}, {1}. Please provide the true "
                             "labels explicitly through the labels argument. "
                             "Classes found in "
                             "y_true: {2}".format(transformed_labels.shape[1],
                                                  y_pred.shape[1],
                                                  lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different '
                             'from that in y_pred. Classes found in '
                             'labels: {0}'.format(lb.classes_))

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    print(transformed_labels)
    print('======================================================')
    print(np.dot(w, np.transpose(X)))
    print('------------------------------------------------------')
    print(y_pred)
    print('******************************************************')
    exit()
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize)

def log_loss_1(w):
    """Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of the true labels given a probabilistic classifier's
    predictions. The log loss is only defined for two or more labels.
    For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is
        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    Read more in the :ref:`User Guide <log_loss>`.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.
    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).
    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    labels : array-like, optional (default=None)
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.
        .. versionadded:: 0.18
    Returns
    -------
    loss : float
    Examples
    --------
    >>> log_loss(["spam", "ham", "ham", "spam"],  # doctest: +ELLIPSIS
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...
    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    # do this hack for hessian computation
    eps=1e-15
    normalize=True
    sample_weight=None
    labels=None
    # define labels here:
#    X = np.array([[0,2], [2, 0]])
#    y_true=np.array([1,0])
    X = np.array([[2, 0]])
    y_true=np.array([-1])
#    y_true=-1
    y_pred= np.dot(w, np.transpose(X))

    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please '
                             'provide the true labels explicitly through the '
                             'labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two '
                             'labels for log_loss, '
                             'got {0}.'.format(lb.classes_))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError("y_true and y_pred contain different number of "
                             "classes {0}, {1}. Please provide the true "
                             "labels explicitly through the labels argument. "
                             "Classes found in "
                             "y_true: {2}".format(transformed_labels.shape[1],
                                                  y_pred.shape[1],
                                                  lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different '
                             'from that in y_pred. Classes found in '
                             'labels: {0}'.format(lb.classes_))

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize)
'''