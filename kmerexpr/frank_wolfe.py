# Exponentiated Gradient Descent
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit
from numpy.linalg import norm
from numpy import maximum, sqrt
from exp_grad_solver import prod_exp_normalize, update_records


def linesearch(x, loss0, d, gd, loss_grad, a_init =1.0, M =1.0, tau =1.2, max_iter =100):
    """
        x: starting point, numpy array in the simplex
        d: search direction
        gd:  grad^T d
    Line search to find stepsize such that 
    Determines a stepsize a such that f(x_new) < f(x)+ a grad^T d +a^2*M/2 ||x_new- x||^2 
    to guarantee ascent
    """
    normdsqr = np.linalg.norm(d)**2
    q = lambda a:  loss0 + a*gd +((M*a**2)/2) *normdsqr
    a = a_init
    q_val = loss0
    loss_new =loss0
    for count in range(max_iter):
        M = M*tau
        a = np.minimum(-gd/(M*normdsqr),a_init)
        if a == a_init:
            M = tau*(-gd/(a_init*normdsqr))
        q_val = q(a)
        x_new = x+a*d
        loss_new =  loss_grad(x_new, nograd =True)
        if loss_new < q_val:
            break
    return a, M


def frank_wolfe_solver(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True,  n = None,  away_step = False, pairsewise_step = False):
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
    loss0, grad0  = loss_grad(x_0)
    grad = grad0.copy()
    loss = loss0
    loss_old = loss
    normg0 = sqrt(grad0 @ grad0)
    norm_records = []
    loss_records = []
    xs = []
    iteration_counts =[]
    num_steps_between_snapshot = maximum(int(n_iters/15),1)
    M  = 0 #initial guess of smoothness
    for iter in range(n_iters):
        imin = np.argmin(grad)  #FW direction
        test_pass = False
        if away_step and x.any() != 0: #away step
            imax = np.argmax(grad[x >0])  #Away step
            test_pass = grad[imax] +grad[imin] <= 2*grad.dot(x)     
        if away_step and test_pass: # away step
            lrst = 0.95*(x[imax]/(1-x[imax]))
            d = x.copy()
            d[imax] = d[imax] -1
        else: # FW step
            lrst = 0.95
            d = -x.copy()
            d[imin] = d[imin] +1

        if type(lrs) == str:
            if away_step and test_pass:
                gd = grad@(x) - grad[imax] 
            else:
                gd = grad[imin] - grad@(x)   # d  =e_max - x
            if M ==0:
                eps= 0.001
                _, gradeps  = loss_grad(x_0+eps*d)
                M = np.linalg.norm(grad -gradeps)/ (eps*np.linalg.norm(d))
            else: 
                eta = 0.5
                M_new = eta*M
                if loss_old != loss:
                    M_new = gd**2/(2*(loss_old-loss)*np.linalg.norm(d)**2)
                M = np.clip(M_new, eta*M, M)
            lrst, M = linesearch(x, loss, d, gd, loss_grad, a_init =lrst, M = M)
        elif lrs is None:
            lrst = 2/(iter+3)
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]
        x_new = x+lrst*d #update
        if np.isnan(x_new.sum()):
            print("iterates have a NaN a iteration ",iter, " existing and return previous iterate" )
            break
        x_av = momentum*x_av +(1-momentum)*x_new
        loss_old = loss
        loss, grad_new  = loss_grad(x_new)
        
        # Checking if method is stopping
        if norm(x_new - x, ord =1) <= tol:
            print("Frank Wolfe iterates are less than: " + str(tol), " apart. Stopping")
            break
        if norm(grad_new -grad , ord =1)<= gtol: 
            print("Frank Wolfe grads are less than: " + str(gtol), " apart. Stopping")
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
