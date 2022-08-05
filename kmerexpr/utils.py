import os
import numpy as np
import pickle

def get_path_prefix_surfix(name, N, L):
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    DATA_PATH = os.path.join(HERE, "..", "data")
    ISO_FILE = os.path.join(DATA_PATH,  name )
    name_split = name.split(".", 1)
    name_no_surfix = name_split[0]
    prefix = os.path.join(DATA_PATH, "model/")
    surfix = name_no_surfix + "-" + str(N) + "-" + str(L) 
    return ISO_FILE, prefix, surfix

def get_path_names(filename, N, L, K, alpha=None):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    READS_FILE = prefix + "READS-" + surfix + ".fna"
    X_FILE = prefix + "X-" + surfix + "-" + str(K) + "_csr.npz"
    Y_FILE = prefix + "Y-" + surfix + "-" + str(K)
    if alpha is None:
        Y_FILE = Y_FILE + ".npy"
        READS_FILE =  READS_FILE + ".fna"
    else:
        Y_FILE = Y_FILE +"-a-" +str(alpha) + ".npy"
        READS_FILE =  READS_FILE +"-a-" +str(alpha) + ".fna"
    return ISO_FILE, READS_FILE, X_FILE, Y_FILE

def save_lengths(filename, N, L, lengths):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-"+str(L)+"-lengths.npy"
    np.save(LENGTHS_FILE,lengths)

def load_lengths(filename, N, L):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    LENGTHS_FILE = prefix + filename + "-L-"+str(L)+"-lengths.npy"
    return np.load(LENGTHS_FILE)

def save_simulation_parameters(filename, N, L, alpha,  psi=None, theta_true=None, theta_sampled=None):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    DICT_FILE = prefix + "SIM_PARAMETERS" + surfix + str(alpha)
    dict = {}
    dict['alpha'] = alpha
    dict['psi'] = psi
    dict['theta_true'] = theta_true
    dict['theta_sampled'] = theta_sampled
    a_file = open(DICT_FILE, "wb")
    pickle.dump(dict, a_file)
    a_file.close()

def load_simulation_parameters(filename, N, L, alpha):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    DICT_FILE = prefix + "SIM_PARAMETERS" + surfix +str(alpha)
    a_file = open(DICT_FILE, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    return dict

def save_run_result(filename, model_name, N, L,  K, dict, alpha=None):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    DICT_FILE = prefix + "DICT-RESULTS-" + model_name + "-" + surfix + str(K)
    if alpha is not None:
        DICT_FILE = DICT_FILE + "-a-" +str(alpha)
    a_file = open(DICT_FILE, "wb")
    pickle.dump(dict, a_file)
    a_file.close()

def load_run_result(filename, model_name, N, L,  K, alpha=None):
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N, L)
    DICT_FILE = prefix + "DICT-RESULTS-" + model_name + "-" + surfix + str(K)
    if alpha is not None:
        DICT_FILE = DICT_FILE + "-a-" +str(alpha)
    try:
        a_file = open(DICT_FILE, "rb")
        dict = pickle.load(a_file)
        a_file.close()
    except:
        dict= None
    return dict
