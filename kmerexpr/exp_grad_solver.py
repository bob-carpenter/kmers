# Exponentiated Gradient Descent
import numpy as np
import time
from scipy.special import softmax as softmax

def prod_exp_normalize(x,y):
    """ Numerically stable implementation of exp_normalize function  """
    # import pdb; pdb.set_trace()
    b = y.max()
    u = x*np.exp(y - b)
    return u / u.sum()

def exp_grad_solver(loss_grad,  x_0, lrs=None, tol=10**(-8.0), gtol = 10**(-8.0),  n_iters = 10000, verbose=True):
    """
    Exponentiated Gradient Descent for minimizing
    max l(x)  s.t. sum(x) = 1 and x>0

    updates take the form of
    x^{t+1}  =  (x^t_i exp( lr_t  grad^t_i )) /(sum (x^t_j exp(lr_t  grad^t_j )).
    """
    x = x_0.copy()
    loss0, grad0  = loss_grad(x_0)
    grad = grad0.copy()
    normg0 = np.sqrt(grad0 @ grad0)
    norm_records = [normg0]
    loss_records = [1.0]

    num_steps_between_snapshot = np.maximum(int(n_iters/15),1)
    num_steps_between_tolerance_check = np.minimum(5, num_steps_between_snapshot)
    for iter in range(n_iters):
        
        if lrs is None:
            lrst = 2**(1/2)/(np.linalg.norm(grad, np.inf)*np.sqrt(iter+1) )
        elif lrs.shape == (1,):
            lrst = lrs[0]
        else:
            lrst = lrs[iter]
        # alternative update using prod exp function
        # x = prod_exp_normalize(x, lrst*grad)
        x_old = x.copy()
        x = softmax(np.log(x) +lrst*grad)  # 
        loss, grad  = loss_grad(x)
        # print(iter, "loss: ", loss)
        # print("theta: ", x)
        if (iter + 1) % num_steps_between_tolerance_check == 0:
            relative_grad_norm = np.sqrt(grad @ grad)/normg0
            relative_loss = loss/loss0
            if np.linalg.norm(x_old - x) <= tol: 
            # if relative_grad_norm <= gtol or relative_loss <= tol: ## Makes no sense now
                print("Exponential grad iterates are less than: " + str(tol), " apart. Stopping")
                break

        if (iter + 1) % num_steps_between_snapshot == 0:
            norm_records.append(relative_grad_norm)
            loss_records.append(relative_loss)
            # Print progress
            if verbose:
                print(" | norm of gradient {:f} | loss {:f} |".format(norm_records[-1],loss_records[-1]))

    norm_records.append(relative_grad_norm)
    loss_records.append(relative_loss)                                                                                              
    return {'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records}