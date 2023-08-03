"""
Reference: https://github.com/oscarknagg/few-shot/blob/master/few_shot/matching.py
Data: 2023/02/13
"""

import torch
import time

EPSILON = 1e-8
loss_fn = torch.nn.NLLLoss().cuda()

def matching_loss(prediction, target, n_support, n_way):
    '''
    Args:
    - prediction: the model output for a batch of samples
    - target: ground truth for batch of samples
    - n-support: number of support samples
    - n-way: number of way
    '''

    prediction_cpu = prediction.to('cpu') # [class*(num_support + num_querry)]
    target_cpu = target.to('cpu') # [class*(num_support + num_querry)]

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support # number of samples per class - n_support

    def find_support_idxs(c):
        """
        Input a class 'c', return the indexes of support samples
        Fetch the first n_support samples as the support set per classes
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    
    def find_query_indxs(c):
        """
        Input a class 'c', return the indexes of query samples
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[n_support:]

    # Get support and query set indexes
    support_indexes = torch.stack(list(map(find_support_idxs, classes))).view(-1)
    query_indexes = torch.stack(list(map(find_query_indxs, classes))).view(-1)

    # Fetch the support and query sample predictions
    support_samples = prediction[support_indexes] # 5-way 5-shot [50,64]
    query_samples = prediction[query_indexes] # 5-way 5-shot [50,64]

    # compute distances
    dists = pairwise_distances(query_samples, support_samples, 'cosine') # [50, 50]

    # Calculate "attention" as softmax over support-query distances
    attention = (-dists).softmax(dim=1) # [50, 50]

    # Calculate predictions as in equation (1) from Matching Networks
    # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
    y_pred = matching_net_predictions(attention, n_support, n_way, n_query) # [50,10]

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    loss = loss_fn(clipped_y_pred.log(), target[query_indexes].to(dtype=torch.int64))

    y_hat = torch.max(y_pred, 1)[1].cpu().numpy() # max prediction indexs
    y_true = target[query_indexes].to(dtype=torch.int64).cpu().numpy()
    acc_query = (y_hat == y_true).sum() / len(y_true)


    return loss, acc_query


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.

    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class

    # Returns
        y_pred: Predicted class probabilities
    """
    if attention.shape != (q * k, k * n): # 10-way 5-shot: [50, 50]
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k) # 10-way 5-shot: [50, 10]

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label(k, n).unsqueeze(-1) # y.shae = [5，1], y = [[0],[1],[2],[3],[4]]
    y_onehot = y_onehot.scatter(1, y, 1)  # 生成onehot标签
    y_pred = torch.mm(attention.to(dtype=torch.float64), y_onehot.cuda().double()) # float64

    return y_pred

def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q

    # TODO: Test this

    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y