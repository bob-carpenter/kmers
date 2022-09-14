
Todo:
1. Write tests for new code (fw, linesearch?)
2. Use line search with frank-wolfe as well
3. Generate a dashboard of plots, with number of reads on x-axis and MSE on y axis, with on curve for each possible k-mer count. Also different plots for alpha that simulates the data.
4. Plot comparison of convergence pf EG and FW with and without line search. The compare scatter plots of best two, and scatter plot of softmax+LBFGS

Clean-ups:
1. Remove failed methods, accel_exp_grad, away step of frank wolfe, mirror_lbfgs
2. Remove obselte parameters and options: Hessinv, continue from last
3. Decide which restart option for linesearch in exp_grad to use
4. remove failed prior in loss_grad 


Ongoing tests:

1. Waiting for Frank wolfe experiments. Trying M = 0.5*M decrease, but its M = 0.8*M on the full genome test
2. Try for last time Hess_inv with exp_grad
3. Try different initialization with Frank_wolfe?
4. Changed the prior/regularizor for the simple log(1/theta) in multinomial_simplex_model.logp_grad. Remove or change back?

BEST RESULT:

simplex, lrs= "lin-warmstart", lrst = 1.3*lrst,  theta0 = self.initialize_iterates_uniform()   


Reserve TODO:

1. Write bfgs with line search that keeps it inside the simplex. Also test BFGS without line search? Should be abble to solve a quadratic with a constant step size.