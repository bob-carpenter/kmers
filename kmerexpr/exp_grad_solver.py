# Exponentiated Gradient Descent
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit

@jit(nopython=True)
def prod_exp_normalize(x,y):
    """ Numerically stable implementation of exp_normalize function  """
    b = y.max()
    u = x*np.exp(y - b)
    return u / u.sum()

def  update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts):
    relative_grad_norm = np.sqrt(grad @ grad)/normg0
    relative_loss = loss/loss0
    xs.append(x)
    norm_records.append(relative_grad_norm)
    loss_records.append(relative_loss)
    iteration_counts.append(iter)

def exp_grad_solver(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True,  batchsize = None, n = None, continue_from = 0):
    """
    Exponentiated Gradient Descent for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (x^t_i exp( lr_t  grad^t_i )) /(sum (x^t_j exp(lr_t  grad^t_j )).

    n = number of sub functions in l(x) = sum_{i=1}^n l_i(x), Assumes l_i(x), nabla l_i(x) = loss_grad(x, i) 
    batchsize = number of 
    """
    x = x_0.copy()
    x_av = x_0.copy()
    momentum = 0.9
    loss0, grad0  = loss_grad(x_0)
    grad = grad0.copy()
    normg0 = np.sqrt(grad0 @ grad0)
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts =[]
    batch = None
    num_steps_between_snapshot = np.maximum(int(n_iters/15),1)
    num_steps_before_decrease =0
    for iter in range(n_iters):
        if lrs is None:
            if iter < num_steps_before_decrease:
                lrst = 2**(1/2)/(np.linalg.norm(grad, np.inf) )
            else:
                lrst = 2**(1/2)/(np.linalg.norm(grad, np.inf)*np.sqrt(iter+1-num_steps_before_decrease+continue_from) )
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]
        # alternative update using prod exp function
        x_new = prod_exp_normalize(x, lrst*grad)
        # x_new = softmax(np.log(x) +lrst*grad)  # 
        x_av = momentum*x_av +(1-momentum)*x_new
        if batchsize is not None:
            batch = np.random.choice(n, batchsize)
        loss, grad_new  = loss_grad(x_new, batch = batch)
        # Checking if method is stopping
        if np.linalg.norm(x_new - x, ord =1) <= tol:
            print("Exp_grad iterates are less than: " + str(tol), " apart. Stopping")
            break
        if np.linalg.norm(grad_new -grad , ord =1)<= gtol: 
            print("Exp_grad grads are less than: " + str(gtol), " apart. Stopping")
            break
        grad = grad_new
        x = x_new
        if (iter + 1) % num_steps_between_snapshot == 0:
            update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
            # Print progress
            if verbose:
                print("iter {:n} | norm of gradient {:f} | loss {:f} |".format(iteration_counts[-1],norm_records[-1],loss_records[-1]))

    update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
    dict_out =  {'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records, 'iteration_counts' : iteration_counts, 'xs' : xs, 'x_av' : x_av}                                                                         
    return dict_out
