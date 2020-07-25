import numpy as np


def precision(ground_truth, prediction):
    """Compute Precision metric.

    Args:
        ground_truth (List): the ground truth set or sequence
        prediction (List): the predicted set or sequence

    Returns:
        precision_score (float): the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    precision_score = count_a_in_b_unique(prediction, ground_truth) / float(
        len(prediction)
    )
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ground_truth, prediction):
    """Compute Recall metric.

    Args:
        ground_truth (List): the ground truth set or sequence
        prediction (List): the predicted set or sequence

    Returns:
        recall_score (float): the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    recall_score = (
        0
        if len(prediction) == 0
        else count_a_in_b_unique(prediction, ground_truth) / float(len(ground_truth))
    )
    assert 0 <= recall_score <= 1
    return recall_score


def mrr(ground_truth, prediction):
    """Compute Mean Reciprocal Rank metric. Reciprocal Rank is set 0 if no predicted item is in contained the ground truth.

    Args:
        ground_truth (List): the ground truth set or sequence
        prediction (List): the predicted set or sequence

    Returns:
        rr (float): the value of the metric
    """
    rr = 0.0
    for rank, p in enumerate(prediction):
        if p in ground_truth:
            rr = 1.0 / (rank + 1)
            break
    return rr


def ndcg(ground_truth, prediction):
    """Compute Normalized Discounted Cumulative Gain (NDCG) metric.

    Args:
        ground_truth (List): the ground truth set or sequence.
        prediction (List): the predicted set or sequence.

    Returns:
        ndcg (float): the value of the metric.
    """
    # ground_truth = remove_duplicates(ground_truth)
    # prediction = remove_duplicates(prediction)
    gt_pos = [1 if i in ground_truth else 0 for i in prediction]
    pd_rank = [rank for rank, i in enumerate(prediction) if i in ground_truth]

    def dcg_score(gt_pos, pd_rank):
        ranked_scores = np.take(gt_pos, pd_rank)
        gain = 2 ** ranked_scores - 1
        discounts = [np.log2(rank + 2) for rank in pd_rank]
        return np.sum(gain / discounts)

    if len(pd_rank) != 0:
        dcg = dcg_score(gt_pos, pd_rank)

        i_gt_pos = [gt_pos[i] for i in np.argsort(gt_pos)[::-1]]
        i_pd_rank = [rank for rank, i in enumerate(i_gt_pos) if i not in [0]]
        idcg = dcg_score(i_gt_pos, i_pd_rank)
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return ndcg


def count_a_in_b_unique(a, b):
    """Count unique items.

    Args:
        a (List): list of lists.
        b (List): list of lists.

    Returns:
        count (int): number of elements of a in b.
    """
    count = 0
    for el in a:
        if el in b:
            count += 1
    return count


def remove_duplicates(li):
    """Remove duplicated items in the list."""
    return [list(x) for x in set(tuple(x) for x in li)]
