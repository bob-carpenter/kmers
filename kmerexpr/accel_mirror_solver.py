# Exponentiated Gradient Descent
from matplotlib import projections
import numpy as np
import time
from scipy.special import softmax as softmax
from numba import jit
from numpy.linalg import norm
from numpy import maximum, sqrt
import thirdparty.amd.AcceleratedMethods  as ac
import thirdparty.amd.Projections as proj
from exp_grad_solver import prod_exp_normalize, update_records

def simplex_clip(x,epsilon):
    x= softmax(x+epsilon)

def accel_mirror_solver(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True,  n = None, Hessinv=False):
    """
    Exponentiated Gradient Descent for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (x^t_i exp( lr_t  grad^t_i )) /(sum (x^t_j exp(lr_t  grad^t_j )).

    n = number of sub functions in l(x) = sum_{i=1}^n l_i(x), Assumes l_i(x), nabla l_i(x) = loss_grad(x, i) 
    batchsize = number of 
    """
    d = x_0.size
    precision = 1e-10
    epsilon = .3
    # Simplex constrained projections
    # psp = proj.SimplexProjectionPNorm(p = 1.9, precision=precision)
    # ps2 = proj.SimplexProjectionEuclidean(precision=precision)
    # ps1 = proj.SimplexProjectionPNorm(p=1.2, precision=precision)
    # psExp = proj.SimplexProjectionExp(dimension = d, precision=precision, epsilon = 1/d)
    psExpS = proj.SimplexProjectionExpSort(dimension = d, epsilon = epsilon)
    psExpS0 = proj.SimplexProjectionExpSort(dimension = d, epsilon = 0)
    # Unconstrained transformation
    # noProj = proj.NoProjection()
    # pExp = proj.PotentialProjectionExp(dimension = d, epsilon = 1/d)


    lmax = 20
    s = 1/lmax
    r = 3
    p1 = psExpS
    p2 = psExpS0
    s1 = s*p1.epsilon/(1+d*p1.epsilon)
    print(s1)
    s2 = s

    
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
    f = lambda theta: -loss_grad(theta)[0]
    gradf = lambda theta: -loss_grad(theta)[1]

    amd = ac.AcceleratedMethodWithSpeedRestart(f, gradf, p1, p2, s1, s2, r, x_0, 'speed restart')
    # amd = ac.AcceleratedMethod(f, gradf, p1, p2, s1, s2, r, x_0, 'accelerated descent')
    # amd = ac.AcceleratedMethodWithRestartFunctionScheme(f, gradf, p1, p2, s1, s2, r, x_0, 'function restart')
    # amd = ac.AcceleratedMethodWithRestartGradScheme(f, gradf, p1, p2, s1, s2, r, x_0, 'gradient restart')
    # amd = ac.MDMethod(f, gradf, p2, s2, x_0, 'mirror descent')
    epsilon = 1e-10
    for iter in range(n_iters):
        if lrs is None:
            lrst = 2**(-1/2)/(norm(grad, np.inf)*sqrt(iter+1) )
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]
        amd.s1 = lrst  # setting the step size
        # import pdb; pdb.set_trace()
        simplex_clip(amd.x,epsilon)
        simplex_clip(amd.z,epsilon)
        amd.step()
        x_new = amd.x
        z_new = amd.z if hasattr(amd, 'z') else amd.x
        # simplex_clip(x_new,epsilon)
        # simplex_clip(z_new,epsilon)
        if np.isnan(x_new.sum()):
            print("iterates have a NaN at iteration",iter, ". Exiting and return previous iterate" )
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
