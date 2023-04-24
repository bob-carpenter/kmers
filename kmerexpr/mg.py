import warnings
from typing import Any, Callable, Dict, Optional, Tuple
from numpy.typing import NDArray
import numpy as np
import time
from numpy.linalg import norm
from kmerexpr.exp_grad_solver import update_records


def mg(logp_grad, param, tol=10 ** (-20.0), max_iter=2000, callback=None):
    """The Multiplicative Gradient algorithm."""
    num_steps_between_snapshot = np.maximum(int(max_iter / 15), 1)
    obj_init, grad_init = logp_grad(param)
    normg0 = np.sqrt(grad_init @ grad_init)
    grad = grad_init.copy()
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts = []
    import pdb

    pdb.set_trace()
    for t in range(max_iter):
        # Update theta = theta . grad
        param = param * grad
        param = param / np.sum(param)
        obj, grad = logp_grad(param)

        if callback is not None:
            callback(param, None)

        if np.isnan(param).any():
            warnings.warn(
                "iterates have a NaN a iteration {iter}; returning previous iterate"
            )
        if (t + 1) % num_steps_between_snapshot == 0:
            update_records(
                grad,
                normg0,
                obj,
                obj_init,
                param,
                t,
                xs,
                norm_records,
                loss_records,
                iteration_counts,
            )

    dict_out = {
        "x": param,
        "norm_records": norm_records,
        "loss_records": loss_records,
        "iteration_counts": iteration_counts,
        "xs": xs,
    }

    return dict_out
