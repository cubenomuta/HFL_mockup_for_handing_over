"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import NDArrays
from flwr.common.typing import NDArray
from models.metric_learning import SpreadoutRegularizer


def aggregate_and_spreadout(results: List[Tuple[NDArrays, int, str]], num_clients: int, num_features: int, nu: float, lr: float) ->Tuple[NDArrays, Dict[str, NDArray]]:
    """Compute weighted average."""
    # Create a classification matrix from class embeddings
    embeddings: NDArray = np.zeros((num_clients,num_features))
    cid_dict: Dict[str, int] = {} 
    embedding_dict: Dict[str, NDArray] = {} 

    for idx, res in enumerate(results):
        weights, _, cid = res
        cid_dict[cid] = idx
        if "ipv4" in cid:
            embeddings[idx,:] = weights[-1]
        else:
            embeddings[int(cid),:] = weights[-1]
    
    embeddings = torch.nn.Parameter(torch.tensor(embeddings))
    regularizer = SpreadoutRegularizer(nu=nu)
    optimizer = torch.optim.SGD([embeddings], lr=lr)
    optimizer.zero_grad()
    loss = regularizer(embeddings, out_dims=num_clients)
    print(loss)
    loss.backward()
    optimizer.step()
    embeddings = F.normalize(embeddings).detach().cpu().numpy()
    
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples, _ in results])

    # Create a list of weights, each multiplied by the related number of examples
    feature_weights = [
        [layer * num_examples for layer in weights[:-1]] for weights, num_examples, _ in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*feature_weights)
    ]
    weights_prime.append(embeddings)
    for cid, idx in cid_dict.items():
        embedding_dict[cid] = embeddings[np.newaxis, idx, :]

    return weights_prime, embedding_dict
