from mimetypes import init
import os
import numpy as np
import pickle
from typing import Any

from dataclasses import dataclass, asdict
import multinomial_model as mm
import multinomial_simplex_model as msm
import normal_model as mnm
import hashlib

@dataclass(frozen=True)
class Problem:
    filename: str
    N: int         # number of reads
    K: int = 15    # length of kmers
    L: int = 100   # length of reads
    alpha: float = 0.1  # parameter of Dirchlet distribution prior for simulating reads

    def get_path_names(self):
        ISO_FILE, prefix, surfix = get_path_prefix_surfix(self.filename, self.N, self.L)
        READS_FILE = prefix + "READS-" + surfix + ".fna"
        X_FILE = prefix + "X-" + surfix + "-" + str(self.K) + "_csr.npz"
        Y_FILE = prefix + "Y-" + surfix + "-" + str(self.K)
        Y_FILE = Y_FILE +"-a-" +str(self.alpha) + ".npy"
        READS_FILE =  READS_FILE +"-a-" +str(self.alpha) + ".fna"
        return ISO_FILE, READS_FILE, X_FILE, Y_FILE
        
@dataclass
class Model_Parameters:
    model_type: str
    solver_name: str = "empty"
    beta: float = 1.0  # parameter for prior over theta
    lrs: Any = None    # options for line search 
    init_iterates: str = "uniform" #options for initialize iterates
    joker: Any = False
    # n_iters: int =2000
    # tol: float=1e-20 
    # gtol: float=1e-20
    # Hessinv = False

    def initialize_model(self, X_FILE, Y_FILE,  lengths):
        if(self.model_type == "softmax"):
            model_class = mm.multinomial_model
        elif(self.model_type == "normal"):
            model_class = mnm.normal_model
        else:
            model_class = msm.multinomial_simplex_model
        return model_class(X_FILE, Y_FILE, beta = self.beta, lengths = lengths, solver_name = self.solver_name)


def get_plot_title(problem, model_parameters):
    title =  problem.filename+'-'+ model_parameters.model_type  + "-N-" + str(problem.N) \
            + "-L-" + str(problem.L) + "-K-"+str(problem.K) +"-init-" +model_parameters.init_iterates
    if model_parameters.solver_name != "empty":
        title =  title + "-" + model_parameters.solver_name
    if model_parameters.lrs != None:
        title = title + '-' + model_parameters.lrs
    return title


def get_path_prefix_surfix(name, N, L):
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    DATA_PATH = os.path.join(HERE, "..", "data")
    ISO_FILE = os.path.join(DATA_PATH,  name )
    name_split = name.split(".", 1)
    name_no_surfix = name_split[0]
    prefix = os.path.join(DATA_PATH, "model/")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    surfix = name_no_surfix + "-" + str(N) + "-" + str(L) 
    return ISO_FILE, prefix, surfix

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def save_lengths(filename, N, L, lengths):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-"+str(L)+"-lengths.npy"
    np.save(LENGTHS_FILE,lengths)

def load_lengths(filename, N, L):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-"+str(L)+"-lengths.npy"
    return np.load(LENGTHS_FILE)

def get_simulation_dir(problem):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(problem.filename, problem.N, problem.L)
    sim_id = hash_dict({"filename" : problem.filename, "N" :problem.N, "L" : problem.L, "alpha" : problem.alpha})
    sim_dir = os.path.join(prefix, "simulations/")
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    sim_dir = sim_dir+str(sim_id)
    return sim_dir

def save_simulation_parameters(problem,  psi=None, theta_true=None, theta_sampled=None):
    dict = {}
    dict['alpha'] = problem.alpha
    dict['psi'] = psi
    dict['theta_true'] = theta_true
    dict['theta_sampled'] = theta_sampled

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
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(problem.filename, problem.N, problem.L)
    exp_dict = Merge(asdict(problem), asdict(model_parameters))
    exp_id = hash_dict(exp_dict)
    exp_dir = os.path.join(prefix, "results/")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp_dir = exp_dir+str(exp_id)
    return exp_dir

def save_run_result(problem, model_parameters,  dict):
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
    errors =[]
    for x in xs:
        errors.append(np.linalg.norm(x - theta_true, ord =1))
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