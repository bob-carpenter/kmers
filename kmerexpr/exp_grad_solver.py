# Exponentiated Gradient Descent
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit
from numpy.linalg import norm
from numpy import maximum, sqrt

def linesearch(x, grad, loss, loss_grad, a_init: float = 1.0, eta: float  =0.9, max_iter: int = 20, armijo = False):
    """
        x: starting point, numpy array in the simplex

    Line search to find stepsize such that 
    Determines a stepsize a such that f(x_new) > f(x)+grad^T(x_new -x) +(1/a)* KL(x_new, x) 
    to guarantee ascent
    """
    #Determine max possible stepsize
    # q = lambda x_new, a:  loss + grad@(x_new -x) -(1/a)*x_new@np.log(x_new/x) + np.sum(x_new-x) 
    a = a_init
    q_val = loss
    loss_new = loss
    x_new = x
    mask = x > 0
    for count in range(max_iter):
        if loss_new > q_val: 
            break
        a = a*eta
        x_new = prod_exp_normalize(x, a*grad)
        if armijo:
            q_val = loss + eta*grad@(x_new -x)
        else:
            q_val = loss + grad@(x_new -x) -(1/a)*x_new[mask]@np.log(x_new[mask]/x[mask])  #q(x_new,a)
        loss_new =  loss_grad(x_new, nograd = True)
        if np.isnan(loss_new): # back up more
            loss_new = loss
            a = a*eta**2
    if loss_new <= q_val: 
        print("LINESEARCH FAILED! Caution")
    return x_new, a

@jit(nopython=True)
def prod_exp_normalize(x,y):
    """ Numerically stable implementation of exp_normalize function  """
    b = y.max()
    u = x*np.exp(y - b)
    return u / u.sum()

def update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts):
    relative_grad_norm = np.sqrt(grad @ grad)/normg0
    xs.append(x)
    norm_records.append(relative_grad_norm)
    loss_records.append(loss)
    iteration_counts.append(iter)

def exp_grad_solver(loss_grad, x_0, lrs=None, tol=10**(-8.0), gtol=10**(-8.0), n_iters=10000, verbose=True, hess_inv=False, callback=None):
    """
    Exponentiated Gradient Descent for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (x^t_i exp( lr_t  grad^t_i )) /(sum (x^t_j exp(lr_t  grad^t_j )).

    Args:
        loss_grad (Callable): Function to compute the loss and gradient.
        x_0 (NDarray): Initial parameters.
        lrs (optional): Learning rate. Either a string ("armijo"), a scalar
            learning rate, a list of learning rates to use at each step.
        tol (float, optional): L1 Tolerance for the iterates stopping condition.
            Stops if the iterates don't move by more than tol in L1 norm.
        gtol (float, optional): L1 Tolerance for the gradient stopping condition.
            Stop if the gradient is smaller than gtol in L1 norm.
        n_iters (int, optional): Total number of iterations.
        verbose (float, optional): Print progress on stdout.
        hess_inv (bool, optional): Whether to precondition the gradient with the
            diagonal of the Hessian.
        callback (Callable[x: NDarray], optional): Function called after each
            iteration, passing the current parameters.
    """
    x = x_0.copy()
    x_av = x_0.copy()
    momentum = 0.9
    loss0, grad0  = loss_grad(x_0, hess_inv)
    loss = loss0
    grad = grad0.copy()
    normg0 = sqrt(grad0 @ grad0)
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts =[]
    num_steps_between_snapshot = maximum(int(n_iters/15),1)
    lrst= 2**(-1/2)/(norm(grad, np.inf) )
    for iter in range(n_iters):
        if type(lrs) == str:
            lrst = 1.2*lrst
            if lrs == "armijo":
                armijo = True
            else:   #use previous lrst to warmstart #"lin-warmstart"
                armijo = False
            x_new, lrst = linesearch(x, grad, loss, loss_grad, a_init =lrst, armijo =armijo)
        else:
            if lrs is None:
                if hess_inv:
                    lrst = 1/(norm(grad, np.inf)*(iter+1)**2 )
                else:
                    lrst = 2**(-1/2)/(norm(grad, np.inf)*sqrt(iter+1) )
            elif lrs.shape == (1,):
                lrst = lrs[0]
            else:
                lrst = lrs[iter]
            # Update using prod exp function
            x_new = prod_exp_normalize(x, lrst*grad)

        if np.isnan(x_new.sum()):
            print("iterates have a NaN a iteration ",iter, " existing and return previous iterate" )
            break
        x_av = momentum*x_av +(1-momentum)*x_new
        loss, grad_new = loss_grad(x_new, Hessinv=hess_inv)

        # Checking if method is stopping
        if norm(x_new - x, ord =1) <= tol:
            print("Exp_grad iterates are less than: " + str(tol), " apart. Stopping")
            break
        if norm(grad_new - grad, ord=1) <= gtol:
            print("Exp_grad grads are less than: " + str(gtol), " apart. Stopping")
            break
        grad = grad_new
        x = x_new

        if callback is not None:
            callback(x)

        if (iter + 1) % num_steps_between_snapshot == 0:
            update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
            # Print progress
            if verbose:
                print("iter {:n} | rel. norm of grad {:f} | loss {:f} |".format(iteration_counts[-1],norm_records[-1],loss_records[-1]))

    update_records(grad,normg0,loss,loss0,x, iter, xs,norm_records,loss_records,iteration_counts)
    dict_out =  {'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records, 'iteration_counts' : iteration_counts, 'xs' : xs, 'x_av' : x_av}                                                                         
    return dict_out
