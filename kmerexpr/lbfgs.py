import numpy as np


def backtrack(n, x, fx, g, loss_grad, d, step, xp, lbfgs_parameters):
	count = 0
	dec = 0.5
	inc = 2.1
	result = {'status':0,'fx':fx,'step':step,'x':x}
	# Compute the initial gradient in the search direction.
	dginit = np.dot(g, d)
	# Make sure that s points to a descent direction.
	if 0 < dginit:
		print('[ERROR] not descent direction')
		result['status'] = -1
		return result
	# The initial value of the objective function. 
	finit = fx
	dgtest = lbfgs_parameters.ftol * dginit

	while True:
		# x = xp
		x = x + d * step;
		# Evaluate the function and gradient values. 
		# this would change g
		fx, g = loss_grad(x)
		fx= -fx; g=-g
		count = count + 1
		# chedck the sufficient decrease condition (Armijo condition).
		if fx > finit + (step * dgtest):
			width = dec
		else:
			# check the wolfe condition
			# now g is the gradient of f(xk + step * d)
			dg = np.dot(g, d)
			if dg < lbfgs_parameters.wolfe * dginit:
				width = inc
			else:
				# check the strong wolfe condition
				if dg > -lbfgs_parameters.wolfe * dginit:
					width = dec
				else:
					result = {'status':0, 'fx':fx, 'step':step, 'x':x}
					return result
		if step < lbfgs_parameters.min_step:
			result['status'] = -1
			return result
		if step > lbfgs_parameters.max_step:
			result['status'] = -1
			return result
		if lbfgs_parameters.max_linesearch <= count:
			result = {'status':0, 'fx':fx, 'step':step, 'x':x}
			return result	
		# update the step		
		step = step * width

class lbfgs_parameters:
	"""the parameters of lbfgs method
		m: the number of corrections to approximate the inverse hessian matrix
		Linesearch: the Linesearch method
		epsilon: epsilon for the convergence test
		ftol: A parameter to control the accuracy of the line search routine.The default value is 1e-4. 
			This parameter should be greaterthan zero and smaller than 0.5.
		wolfe: A coefficient for the Wolfe condition.The default value is 0.9. 
			This parameter should be greater the ftol parameter and smaller than 1.0.
		min_step: The minimum step of the line search routine.
		max_step: The maximum step of the line search routine."""

	def __init__(self,  m=10, Linesearch=backtrack, epsilon=1e-5, ftol=1e-4, wolfe=0.9, max_linesearch=10, min_step=1e-20, max_step=1e20):
		self.m = m
		self.Linesearch = Linesearch
		self.epsilon = epsilon
		self.ftol = ftol
		self.wolfe = wolfe
		self.max_linesearch = max_linesearch
		self.min_step = min_step
		self.max_step = max_step

		
class iterationData:
	"""docstring for iterationData"""
	def __init__(self, alpha, s, y, ys):
		self.alpha = alpha
		self.s = s
		self.y = y
		self.ys = ys
		
class lbfgs:
	"""the class of lbfgs method"""
	def __init__(self, n, x):
		self.n = n
		self.x = x
		self.lbfgs_parameters = lbfgs_parameters()
		# initialize the iteration data list which size is specified in the lbfgs_parameters
		lm = []
		for i in np.arange(0, self.lbfgs_parameters.m):
			s = np.zeros(self.n)
			y = np.zeros(self.n)
			lm.append(iterationData(0.0, s, y, 0.0))
		self.lm = lm  #list of differences of iterates and gradients


	def step(self, x, xp, fx, g, gp, loss_grad):
		# d: store the negative gradient of the object function on point x.
		# set d which is the negative gradient
		# compute the initial step
		if (x ==xp).all():
			return -g.copy()
	
		# update vectors s and y:
		s = x - xp
		y = g - gp
		# Compute scalars ys and yy:
		ys = np.dot(y, s)
		yy = np.dot(y, y)
		it = iterationData(0.0, s, y, ys)

		self.lm.append(it)   #Add new pairs for secant equation
		self.lm.pop(0)		 #Revmove oldest pairs for secant equation
		##
		# Recursive formula to compute dir = (H . g).
		d = g.copy()
		for it in reversed(self.lm):
			if(it.ys ==0):
				continue
			it.alpha = np.dot(it.s, d) / it.ys
			d = d - (it.y * it.alpha)
		d = d * (ys/yy)
		for it in self.lm:
			if(it.ys ==0):
				continue
			beta = np.dot(it.y, d) / it.ys
			d = d + (it.s * (it.alpha - beta))
		## Linesearch needs access to function values.
		step =1.0
		# step = 1.0 / np.sqrt(np.dot(d, d))
		# ls = self.lbfgs_parameters.Linesearch(self.n, x, fx, g,loss_grad, -d, step, xp, self.lbfgs_parameters)

		# if ls['status'] < 0:
		# 	x = xp.copy()
		# 	g = gp.copy()
		# 	print('[ERROR] the point return to the previous point')
		# 	return ls['status']
		# step = ls['step']
		# print("step:  ", step)
		return -d*step   #output of ascent direction H . g 