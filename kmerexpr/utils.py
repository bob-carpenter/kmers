import os
import numpy as np
import pickle
from typing import Any

from dataclasses import dataclass, asdict
from kmerexpr import multinomial_model as mm
from kmerexpr import multinomial_simplex_model as msm
import hashlib

from numba import njit, prange
from sparse_dot_mkl import dot_product_mkl

from contextlib import contextmanager
from ctypes import CDLL, c_int
from ctypes.util import find_library

_libmkl_rt_path = find_library('mkl_rt')
_libmkl_rt = CDLL(_libmkl_rt_path)

_libmkl_rt.MKL_Set_Interface_Layer
_libmkl_rt.MKL_Set_Interface_Layer.restype = c_int
_libmkl_rt.MKL_Set_Interface_Layer.argtypes = [c_int]


@njit(parallel=True, fastmath=True)
def _a_dot_logb(a, b):
    res = 0.0
    for i in prange(len(a)):
        res += a[i] * np.log(b[i])

    return res

@njit(parallel=True, fastmath=True)
def _zero(arr):
    for i in prange(len(arr)):
        arr[i] = 0.


@njit(parallel=True, fastmath=True)
def _divide(a, b, c):
    for i in prange(len(a)):
        c[i] = a[i] / b[i]


@contextmanager
def mkl_interface(index_type=np.int64):
    assert index_type == np.int64 or index_type == np.int32
    # LP64 if int32 indices, else ILP64
    index_code = 0 if index_type == np.int32 else 1
    old_interface = _get_interface_layer()
    if old_interface >= 2:  # GNU
        index_code += 2

    _libmkl_rt.MKL_Set_Interface_Layer(index_code)
    yield
    _libmkl_rt.MKL_Set_Interface_Layer(old_interface)


def _get_interface_layer() -> str:
    code: int = 0
    env: str = os.environ.get('MKL_INTERFACE_LAYER', "")
    if 'ILP64' in env:
        code += 1
    if 'GNU' in env:
        code += 2
    return code


def logp_grad(theta, beta, xnnz, ynnz, scratch, lengths, nograd=False):
    """Return negative log density and its gradient evaluated at the specified simplex.

       loss(theta) = y' log(X'theta) + (beta-1 )(sum(log(theta)) - log sum (theta/Lenghts))
       grad(theta) = X y diag{X' theta}^{-1}+ (beta-1 ) (1 - (1/Lenghts)/sum (theta/Lenghts) )
    Keyword arguments:
    theta -- simplex of expected isoform proportions
    """
    theta = theta.astype(np.float32)
    mask = theta > 0
    thetamask = theta[mask]
    scratch.resize(xnnz.shape[0])

    xthetannz = scratch
    _zero(xthetannz)
    with mkl_interface(xnnz.indices.dtype):
        dot_product_mkl(xnnz, theta, out=xthetannz)
        val = _a_dot_logb(ynnz, scratch)
        if beta != 1.0:
            val += (beta - 1.0) * np.sum(np.log(thetamask / lengths[mask]))
            val -= (beta - 1.0) * np.log(np.sum(thetamask / lengths[mask]))

        if nograd:
            return val

        yxTtheta = scratch
        _divide(ynnz, yxTtheta, yxTtheta)
        grad = dot_product_mkl(yxTtheta, xnnz)
        if beta != 1.0:
            grad[mask] += (beta - 1.0) / thetamask
            grad[mask] -= (beta - 1.0) / (np.sum(thetamask / lengths[mask]) * lengths[mask])

        return val, grad


@dataclass(frozen=True)
class Problem:
    filename: str
    N: int = 1000  # number of reads
    K: int = 15  # length of kmers
    L: int = 100  # length of reads
    alpha: float = 0.1  # parameter of Dirchlet distribution prior for simulating reads

    def get_path_names(self):
        ISO_FILE, prefix, surfix = get_path_prefix_surfix(self.filename, self.N, self.L)
        READS_FILE = prefix + "READS-" + surfix + ".fna"
        X_FILE = prefix + "X-" + surfix + "-" + str(self.K) + "_csr.npz"
        Y_FILE = prefix + "Y-" + surfix + "-" + str(self.K)
        Y_FILE = Y_FILE + "-a-" + str(self.alpha) + ".npz"
        READS_FILE = READS_FILE + "-a-" + str(self.alpha) + ".fna"
        return ISO_FILE, READS_FILE, X_FILE, Y_FILE


@dataclass
class Model_Parameters:
    model_type: str
    solver_name: str = "empty"
    beta: float = 1.0  # parameter for prior over theta
    lrs: Any = None  # options for line search
    init_iterates: str = "uniform"  # options for initialize iterates
    joker: Any = False

    def __post_init__(self):
        if self.solver_name != "frank_wolfe" and self.solver_name != "exp_grad":
            print("No solver called", self.solver_name, ". Defaulting to exp_grad")
            self.solver_name = "exp_grad"

    def initialize_model(self, X_FILE, Y_FILE, lengths=None):
        if self.model_type == "softmax":
            model_class = mm.multinomial_model
        else:
            model_class = msm.multinomial_simplex_model
        return model_class(
            X_FILE,
            Y_FILE,
            beta=self.beta,
            lengths=lengths,
            solver_name=self.solver_name,
        )


def get_plot_title(problem, model_parameters):
    title = (
        problem.filename
        + "-"
        + model_parameters.model_type
        + "-N-"
        + str(problem.N)
        + "-L-"
        + str(problem.L)
        + "-K-"
        + str(problem.K)
        + "-init-"
        + model_parameters.init_iterates
        + "-a-"
        + str(problem.alpha)
    )
    if model_parameters.solver_name != "empty":
        title = title + "-" + model_parameters.solver_name
    if model_parameters.lrs != None:
        title = title + "-" + model_parameters.lrs
    return title


def get_path_prefix_surfix(name, N, L):
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    DATA_PATH = os.path.join(HERE, "..", "data")
    ISO_FILE = os.path.join(DATA_PATH, name)
    name_no_dot = name.replace(".", "-")
    prefix = os.path.join(DATA_PATH, "model/")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    surfix = name_no_dot + "-" + str(N) + "-" + str(L)
    return ISO_FILE, prefix, surfix


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def save_lengths(filename, N, L, lengths):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-" + str(L) + "-lengths.npy"
    np.save(LENGTHS_FILE, lengths)


def load_lengths(filename, N, L):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-" + str(L) + "-lengths.npy"
    return np.load(LENGTHS_FILE)


def get_simulation_dir(problem):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(
        problem.filename, problem.N, problem.L
    )
    sim_id = hash_dict(
        {
            "filename": problem.filename,
            "N": problem.N,
            "L": problem.L,
            "alpha": problem.alpha,
        }
    )
    sim_dir = os.path.join(prefix, "simulations/")
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    sim_dir = sim_dir + str(sim_id)
    return sim_dir


def save_simulation_parameters(problem, psi=None, theta_true=None, theta_sampled=None):
    dict = {}
    dict["alpha"] = problem.alpha
    dict["psi"] = psi
    dict["theta_true"] = theta_true
    dict["theta_sampled"] = theta_sampled

    sim_dir = get_simulation_dir(problem)
    a_file = open(sim_dir, "wb")
    pickle.dump(dict, a_file)
    a_file.close()


def load_simulation_parameters(problem):
    sim_dir = get_simulation_dir(problem)
    a_file = open(sim_dir, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    return dict


def get_exp_dir(problem, model_parameters):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(
        problem.filename, problem.N, problem.L
    )
    exp_dict = Merge(asdict(problem), asdict(model_parameters))
    exp_id = hash_dict(exp_dict)
    exp_dir = os.path.join(prefix, "results/")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp_dir = exp_dir + str(exp_id)
    return exp_dir


def save_run_result(problem, model_parameters, dict):
    exp_dir = get_exp_dir(problem, model_parameters)
    a_file = open(exp_dir, "wb")
    pickle.dump(dict, a_file)
    a_file.close()


def load_run_result(problem, model_parameters):
    exp_dir = get_exp_dir(problem, model_parameters)
    a_file = open(exp_dir, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    return dict


def get_errors(xs, theta_true):
    errors = []
    for x in xs:
        errors.append(np.linalg.norm(x - theta_true, ord=1))
    return errors


# from haven-ai
def hash_dict(dictionary):
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()
