# Exponentiated Gradient Descent
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit
from numpy.linalg import norm
from numpy import maximum, sqrt
from exp_grad_solver import prod_exp_normalize, update_records


def linesearch(x, d, gd, loss, loss_grad, a_init =1, tau =1.1):
    """
        x: starting point, numpy array in the simplex
        d: search direction
        gd:  grad^T d
    Line search to find stepsize such that 
    Determines a stepsize a such that f(x_new) < f(x)+ a grad^T d +a^2*M/2 ||x_new- x||^2 
    to guarantee ascent
    """
    #Determine max possible stepsize
    normdsqr = np.linalg.norm(d)**2
    q = lambda a:  loss + a*gd +(M*a**2)/2 *normdsqr
    a = a_init
    q_val = loss
    while loss <= q_val: 
        M = M*tau
        a = np.minimum(-gd/(M*normdsqr),1)
        q_val = q(a)
    return x+a*d, a



def frank_wolfe_solver(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True,  n = None, Hessinv=False, continue_from = 0):
    """
    Frank Wolfe for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (1-gamma)x^t +gamma e_i,

    where i is the largest coordinate of the gradient nabla f(x^k)

    n = number of sub functions in l(x) = sum_{i=1}^n l_i(x), Assumes l_i(x), nabla l_i(x) = loss_grad(x, i) 
    batchsize = number of 
    """
    x = x_0.copy()
    x_av = x_0.copy()
    momentum = 0.9
    loss0, grad0  = loss_grad(x_0, Hessinv)
    grad = grad0.copy()
    normg0 = sqrt(grad0 @ grad0)
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts =[]
    num_steps_between_snapshot = maximum(int(n_iters/15),1)
    active_set = np.zeros(x.shape, dtype=bool)
    active_set_non_empty = False
    for iter in range(n_iters):
        if lrs is None or lrs == "linesearch":
            lrst = 2/(iter+3)
        # elif lrs == "linesearch":
        #     lrst
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]

        imax = np.argmax(grad)  #FW direction
        test1 = True
        if active_set_non_empty: #away step
            imin = np.argmax(grad[active_set])  #Away step
            test1 = grad[imax] +grad[imin] <= 2*grad.dot(x)
        test1 = True        
 
        # test = grad.dot(dFW) >= grad.dot(dA)
        # test1 = True
        
        if test1: # FW step
            x_new = (1-lrst)*x
            x_new[imax] += lrst
        else: # away step
            lrst = lrst*(x[imin]/(1-x[imin]))
            x_new = (1+lrst)*x
            x_new[imin] -= lrst

        active_set = x_new >0
        active_set_non_empty =True
        # if x_new[imax] >0:
        #     active_set[imax] = True
        #     active_set_non_empty =True
        if np.isnan(x_new.sum()):
            print("iterates have a NaN a iteration ",iter, " existing and return previous iterate" )
            break
        x_av = momentum*x_av +(1-momentum)*x_new

        loss, grad_new  = loss_grad(x_new, Hessinv = Hessinv)
        
        # Checking if method is stopping
        if norm(x_new - x, ord =1) <= tol:
            print("Exp_grad iterates are less than: " + str(tol), " apart. Stopping")
            break
        if norm(grad_new -grad , ord =1)<= gtol: 
            print("Exp_grad grads are less than: " + str(gtol), " apart. Stopping")
            break
        grad = grad_new
        x = x_new
        if (iter + 1) % num_steps_between_snapshot == 0:
            update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
            # Print progress
            if verbose:
                print("iter {:n} | rel. norm of grad {:f} | loss {:f} |".format(iteration_counts[-1],norm_records[-1],loss_records[-1]))

    update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
    dict_out =  {'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records, 'iteration_counts' : iteration_counts, 'xs' : xs, 'x_av' : x_av}                                                                         
    return dict_out
