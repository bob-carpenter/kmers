import warnings
import numpy as np
from kmerexpr.exp_grad_solver import update_records
import time

## Code copied from Fred Kunster's kmer_solver_comparison

def mg(logp_grad, param, tol=1E-20, max_iter=2000, callback=None, verbose=True):
    """The Multiplicative Gradient algorithm."""
    num_steps_between_snapshot = np.maximum(int(max_iter / 15), 1)
    obj_init, grad_init = logp_grad(param)
    normg0 = np.sqrt(grad_init @ grad_init)
    grad = grad_init.copy()
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts = []

    st = time.time()
    for t in range(max_iter):
        # Update theta = theta * grad  
        # Note: Removed normalization param = param / np.sum(param) since now expect ynnz to be normalized
        param = param * grad

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
            if verbose:
                dt_per_iter = 1000 * (time.time() - st) / num_steps_between_snapshot
                print(
                    "iter {:n} | rel. norm of grad {:f} | loss {:f} | dt/iter (ms) {:f}".format(
                        iteration_counts[-1], norm_records[-1], loss_records[-1], dt_per_iter
                    )
                )
                st = time.time()

    dict_out = {
        "x": param,
        "norm_records": norm_records,
        "loss_records": loss_records,
        "iteration_counts": iteration_counts,
        "xs": xs,
    }

    return dict_out
