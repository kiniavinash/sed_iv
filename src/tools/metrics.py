# code source: https://github.com/dr-costas/dnd-sed/tree/master/tools/metrics.py

import torch
import pandas as pd
import numpy as np

from .settings import REF_LABELS
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score

_eps: float = torch.finfo(torch.float32).eps


def get_weak_labels(y_hat, mode="multi_class"):
    """
    Aggregates strong labels to weak labels per frame

    :param y_hat: strong labels
    :return: y_hat_weak: weak labels
    """
    if mode == "multi_class":
        # compute idx of max probability for each time step
        _, max_idxs = y_hat.max(dim=1)

        # generate mask
        y_preds = torch.zeros(y_hat.shape)
        y_preds = y_preds.scatter_(1, max_idxs.unsqueeze(1), 1.)

        # majority voting
        sum_votes = y_preds.sum(dim=2)

        ref_labels = REF_LABELS

        # generate weak labels
        _, preds = sum_votes.max(dim=1)
        y_hat_weak = [ref_labels[idx] for idx in preds]

        return y_hat_weak

    elif mode == "single_class":
        # threshold to determine positive or not
        y_hat = y_hat.ge(0.5).float()

        # convert to weak labels by majority voting
        votes = y_hat.sum(dim=2) >= y_hat.shape[2] / 2

        y_hat_weak = [REF_LABELS[0] if vote else REF_LABELS[1] for vote in votes]

        return y_hat_weak


def get_per_class_acc(C):
    pc_accuracies = []

    for i in range(len(C)):
        TP = C[i, i]
        col_sum = np.sum(C[:, i])
        row_sum = np.sum(C[i, :])
        FP = col_sum - TP
        FN = row_sum - TP
        TN = np.sum(C) - TP - FP - FN

        if (TP + TN + FP + FN) == 0:
            pc_accuracies.append(1.0)
        else:
            pc_accuracies.append((TP + TN) / (TP + TN + FP + FN))

    return pc_accuracies


def get_per_class_iou(C):
    pc_accuracies = []

    for i in range(len(C)):
        TP = C[i, i]
        col_sum = np.sum(C[:, i])
        row_sum = np.sum(C[i, :])
        FP = col_sum - TP
        FN = row_sum - TP
        TN = np.sum(C) - TP - FP - FN

        if (TP + FP + FN) == 0:
            pc_accuracies.append(1.0)
        else:
            pc_accuracies.append(TP / (TP + FP + FN))

    return pc_accuracies


def weak_label_metrics(y_hat, y_true, verbose=True, mode="multi_class"):
    y_hat_weak = get_weak_labels(y_hat, mode=mode)
    y_true_weak = get_weak_labels(y_true, mode=mode)

    # compute confusion matrix
    conf_mat_uf = confusion_matrix(y_true_weak, y_hat_weak, labels=REF_LABELS)

    # format matrix using pandas for better readability when printing
    conf_mat = pd.DataFrame(conf_mat_uf,
                            index=['True:' + lbl for lbl in REF_LABELS],
                            columns=['Pred:' + lbl for lbl in REF_LABELS])

    # compute overall accuracy
    accuracy = accuracy_score(y_true_weak, y_hat_weak)

    # compute Jaccard Index per class
    per_class_iou_uf = get_per_class_iou(conf_mat_uf)
    per_class_iou = pd.DataFrame(per_class_iou_uf,
                                 columns=["Per Class IoU"],
                                 index=REF_LABELS)

    # # compute per class accuracy
    # per_class_acc_uf = get_per_class_acc(conf_mat_uf)
    # per_class_acc = pd.DataFrame(per_class_acc_uf,
    #                              columns=["Per Class Accuracy"],
    #                              index=REF_LABELS)

    if verbose:
        print("Confusion Matrix : ")
        print(conf_mat)
        print("=======")

        print("Overall Accuracy: {}".format(accuracy))
        print("=======")

        # print("{}".format(per_class_acc))
        # print("========")

        print("{}".format(per_class_iou))
        print("========")

    return conf_mat, accuracy


def f1_per_frame(y_hat, y_true):
    """Gets the average per frame F1 score, based on\
    TP, FP, and FN, calculated from the `y_hat`\
    predictions and `y_true` ground truth values.

    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :return: F1 score per frame
    :rtype: torch.Tensor
    """
    tp, _, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=None)

    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    the_f1 = _f1(tp=tp, fp=fp, fn=fn)

    return the_f1


def error_rate_per_frame(y_hat, y_true):
    """Calculates the error rate based on FN and FP,
    for one second.
    :param y_hat: Predictions.
    :type y_hat: torch.Tensor
    :param y_true: Ground truth.
    :type y_true: torch.Tensor
    :return: Error rate.
    :rtype: torch.Tensor
    """
    _, __, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=-1)

    s = fn.min(fp).sum()
    d = fn.sub(fp).clamp_min(0).sum()
    i = fp.sub(fn).clamp_min(0).sum()
    n = y_true.sum() + _eps

    return (s + d + i) / n


def _f1(tp, fp, fn):
    """Gets the F1 score from the TP, FP, and FN.
    :param tp: TP
    :type tp: torch.Tensor
    :param fp: FP
    :type fp: torch.Tensor
    :param fn: FN
    :type fn: torch.Tensor
    :return: F1 score
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fp, fn]]):
        return torch.zeros(1).to(tp.device)

    f1_nominator = tp.mul(2)
    f1_denominator = tp.mul(2).add(fn).add(fp)

    return f1_nominator.div(f1_denominator + _eps)


def _tp_tf_fp_fn(y_hat, y_true, dim_sum):
    """Gets the true positive (TP), true negative (TN),\
    false positive (FP), and false negative (FN).
    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :param dim_sum: Dimension to sum TP, TN, FP, and FN. If\
                    it is None, then the default behaviour from\
                    PyTorch`s sum is assumed.
    :type dim_sum: int|None
    :return: TP, TN, FP, FN.
    :rtype: (torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor)
    """
    y_hat_positive = y_hat.ge(0.5)
    y_hat_negative = y_hat.lt(0.5)

    y_true_positive = y_true.eq(1.)
    y_true_negative = y_true.eq(0.)

    tp = y_hat_positive.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    tn = y_hat_negative.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fp = y_hat_positive.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fn = y_hat_negative.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    if dim_sum is not None:
        tp = tp.sum(dim=dim_sum)
        tn = tn.sum(dim=dim_sum)
        fp = fp.sum(dim=dim_sum)
        fn = fn.sum(dim=dim_sum)

    return tp, tn, fp, fn
