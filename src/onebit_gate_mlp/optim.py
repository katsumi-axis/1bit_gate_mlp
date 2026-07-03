from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def split_tree(tree: dict, predicate: Callable[[str], bool]) -> tuple[dict, dict]:
    yes = []
    no = []
    for path, value in tree_flatten(tree):
        if predicate(path):
            yes.append((path, value))
        else:
            no.append((path, value))
    return tree_unflatten(yes), tree_unflatten(no)


class Bop:
    """Binarized optimizer without latent shadow weights.

    The parameter itself is always interpreted as the binary weight. State only
    contains the gradient exponential moving average used for flip decisions.
    """

    def __init__(self, threshold: float = 1e-4, gamma: float = 1e-4):
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if not 0 <= gamma < 1:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        self.threshold = threshold
        self.gamma = gamma
        self.state: dict = {}
        self.last_flip_rate = 0.0

    def update(self, params: dict, grads: dict) -> dict:
        if not self.state:
            self.state = tree_unflatten(
                [(path, mx.zeros_like(value)) for path, value in tree_flatten(params)]
            )

        updates = []
        next_state = []
        flip_count = 0
        param_count = 0
        flat_params = dict(tree_flatten(params))
        flat_state = dict(tree_flatten(self.state))

        for path, grad in tree_flatten(grads):
            weight = mx.sign(flat_params[path])
            momentum = (1.0 - self.gamma) * flat_state[path] + self.gamma * grad
            flip = (mx.abs(momentum) >= self.threshold) & (mx.sign(momentum) == weight)
            next_weight = mx.where(flip, -weight, weight)
            updates.append((path, next_weight))
            next_state.append((path, momentum))
            flip_count += int(mx.sum(flip).item())
            param_count += math.prod(weight.shape)

        self.state = tree_unflatten(next_state)
        self.last_flip_rate = flip_count / param_count if param_count else 0.0
        return tree_unflatten(updates)
