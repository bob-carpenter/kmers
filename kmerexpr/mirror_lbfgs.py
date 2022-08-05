# Mirror BFGS method
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit
from numpy.linalg import norm
from numpy import maximum, sqrt

from lbfgs import lbfgs
from exp_grad_solver import prod_exp_normalize, update_records

def mirror_lbfgs(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True,  batchsize = None, n = None):
    """
    Mirror LBFGS for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (x^t_i exp( lr_t  d^t_i )) /(sum (x^t_j exp(lr_t  d^t_j )),
    where d^t is the descent direction of the LBFGS method

    n = number of sub functions in l(x) = sum_{i=1}^n l_i(x), Assumes l_i(x), nabla l_i(x) = loss_grad(x, i) 
    batchsize = number of 
    """
    x = x_0.copy()
    xp = x_0.copy()
    loss0, grad0  = loss_grad(x_0)
    grad0= -grad0
    fx = loss0
    g = grad0.copy()
    gp = grad0.copy()

    normg0 = sqrt(grad0 @ grad0)
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts =[]
    num_steps_between_snapshot = maximum(int(n_iters/15),1)
    # Initializing lbfgs solver
    lbfgs_solver = lbfgs(x.size, x)
    for iter in range(n_iters):
        # alternative update using prod exp function
        d = lbfgs_solver.step(x, xp, fx, g, gp, loss_grad)
        if lrs is None:
            lrst = 0.5/(norm(d, np.inf)*np.sqrt(iter+1) )
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]
        xp = x.copy()
        x = prod_exp_normalize(x, lrst*d)
        # x = softmax(np.log(x) +lrst*d)  # 
        if np.isnan(x.sum()):
            print("iterates have a NaN a iteration ",iter, " existing and return previous iterate" )
            break

        gp = g.copy()
        fx, g  = loss_grad(x)
        g = -g

        # Checking if method is stopping
        if norm(x - xp, ord =1) <= tol:
            print("Exp_grad iterates are less than: " + str(tol), " apart. Stopping")
            break
        if norm(g -gp , ord =1)<= gtol: 
            print("Exp_grad grads are less than: " + str(gtol), " apart. Stopping")
            break
  
        if (iter + 1) % num_steps_between_snapshot == 0:
            update_records(g,normg0,fx,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
            if verbose: # Print progress
                print("iter {:n} | rel. norm of grad {:f} | loss {:f} |".format(iteration_counts[-1],norm_records[-1],loss_records[-1]))

    update_records(g,normg0,fx,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
    dict_out =  {'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records, 'iteration_counts' : iteration_counts, 'xs' : xs}                                                                         
    return dict_out
