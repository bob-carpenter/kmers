import numpy as np


def backtrack(n, x, fx, g, d, step, xp, lbfgs_parameters):
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
		x = xp
		x = x + d * step;
		# Evaluate the function and gradient values. 
		# this would change g
		fx = lbfgs_parameters.proc_evaluate(x, g, n, step)
		print("[INFO]end line evaluate fx = %r step = %r." %(fx, step))
		count = count + 1
		# chedck the sufficient decrease condition (Armijo condition).
		if fx > finit + (step * dgtest):
			print("[INFO]not satisfy sufficient decrease condition.")
			width = dec
		else:
			# check the wolfe condition
			# now g is the gradient of f(xk + step * d)
			dg = np.dot(g, d)
			if dg < lbfgs_parameters.wolfe * dginit:
				print("[INFO]dg = %r < lbfgs_parameters.wolfe * dginit = %r" %(dg, lbfgs_parameters.wolfe * dginit))
				print("[INFO]not satisfy wolf condition.")
				width = inc
			else:
				# check the strong wolfe condition
				if dg > -lbfgs_parameters.wolfe * dginit:
					print("[INFO]not satisfy strong wolf condition.")
					width = dec
				else:
					result = {'status':0, 'fx':fx, 'step':step, 'x':x}
					return result
		if step < lbfgs_parameters.min_step:
			result['status'] = -1
			print('[ERROR] the linesearch step is too small')
			return result
		if step > lbfgs_parameters.max_step:
			result['status'] = -1
			print('[ERROR] the linesearch step is too large')
			return result
		if lbfgs_parameters.max_linesearch <= count:
			print('[INFO] the iteration of linesearch is many')
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

	def __init__(self, proc_evaluate, proc_progress, m=10, Linesearch=backtrack, epsilon=1e-5, ftol=1e-4, wolfe=0.9, max_linesearch=10, min_step=1e-20, max_step=1e20):
		self.m = m
		self.proc_evaluate = proc_evaluate
		self.proc_progress = proc_progress
		self.Linesearch = Linesearch
		self.epsilon = epsilon
		self.ftol = ftol
		self.wolfe = wolfe
		self.max_linesearch = max_linesearch
		self.min_step = min_step
		self.max_step = max_step

class callbackData:
	"""docstring for callbackData"""
	def __init__(self, n, proc_progress, proc_evaluate):
		self.n = n
		self.proc_progress = proc_progress
		self.proc_evaluate = proc_evaluate
		
class iterationData:
	"""docstring for iterationData"""
	def __init__(self, alpha, s, y, ys):
		self.alpha = alpha
		self.s = s
		self.y = y
		self.ys = ys
		
class lbfgs:
	"""the class of lbfgs method"""
	def __init__(self, n, x, ptr_fx, lbfgs_parameters):
		self.n = n
		self.x = x
		self.ptr_fx = ptr_fx
		self.proc_progress = lbfgs_parameters.proc_progress
		self.proc_evaluate = lbfgs_parameters.proc_evaluate
		self.lbfgs_parameters = lbfgs_parameters

	def initialize_lbfgs(self):
    		ret = 0
		m = self.lbfgs_parameters.m;
		# g: store the gradient of the object function on point x.
		# g = np.array([0.0 for i in np.range(self.n)])

		x = self.x

		# initialize the iteration data list which size is specified in the lbfgs_parameters
		lm = []
		for i in np.range(0, m):
			s = np.array([0.0 for i in np.range(self.n)])
			y = np.array([0.0 for i in np.range(self.n)])
			lm.append(iterationData(0.0, s, y, 0.0))

		return m, x, lm

	def do_lbfgs(self):
		m, g, w, x, lm = self.initialize_lbfgs(self)
		# Evaluate the function value and its gradient. this will change g vector
		fx = self.proc_evaluate(x, g, self.n, 0.)
		print('[INFO]in lbfgs the fx is %r' %(fx))
		# d: store the negative gradient of the object function on point x.
		# set d which is the negative gradient
		d = -g
		# normalize the x vector
		xnorm = np.sqrt(np.dot(x, x))
		# normalize the gradient
		gnorm = np.sqrt(np.dot(g, g))

		if xnorm < 1.0:
			xnorm = 1.0
		# Make sure that the initial variables are not a minimizer.
		if gnorm / xnorm <= self.lbfgs_parameters.epsilon:
			print('[INFO] already at minimizer')
			return 0
		# compute the initial step
		# step = 1.0 / np.sqrt(np.dot(d, d))
		step = 1.0 / np.sqrt(np.dot(d, d))

		k =1
		end = 0
		while True:
			# xp: store the privious point x.
			xp = x.copy()
			# gp: store the privious gradient.
			gp = g.copy()
			# ls should include the status code, fx and step 
			# because the integer parameter of the function is immutable 
			ls = self.lbfgs_parameters.Linesearch(self.n, x, fx, g, d, step, xp, gp, w, self.lbfgs_parameters)

			# revert to the previous point
			if ls['status'] < 0:
				x = xp.copy()
				g = gp.copy()
				print('[ERROR] the point return to the previous point')
				return ls['status']
			fx = ls['fx']
			step = ls['step']
			x = ls['x']

			# compute x and g norms
			xnorm = np.sqrt(np.dot(x, x))
			gnorm = np.sqrt(np.dot(g, g))

			# report the progress
			self.proc_progress(x, g, fx, xnorm, gnorm, step, self.n, k, ls)

			# convergence test.
			if xnorm < 1.0:
				xnorm = 1.0
			if gnorm / xnorm <= self.lbfgs_parameters.epsilon:
				print('[INFO] complete lbfgs')
				return 0

			# update vectors s and y:
			it = lm[end]
			it.s = x - xp
			it.y = g - gp
			# Compute scalars ys and yy:
			ys = np.dot(it.y, it.s)
			yy = np.dot(it.y, it.y)
			it.ys = ys
			##
			# Recursive formula to compute dir = -(H \cdot g).
			# This is described in page 779 of:
			# Jorge Nocedal.
			# Updating Quasi-Newton Matrices with Limited Storage.
			# Mathematics of Computation, Vol. 35, No. 151,
			# pp. 773--782, 1980.
			##
			bound = (m <= k and [m] or [k])[0]
			k = k + 1
			# get the next one in the last m iteration data list
			end = (end + 1) % m

			# compute the steepst direction
			# compute the negative gradients
			d = -g
			j = end

			for i in np.range(0, bound):
				# from later to former
				j = (j + m -1) % m
				it = lm[j]
				it.alpha = np.dot(it.s, d) / it.ys
				d = d + (it.y * (-it.alpha))

			d = d * (ys/yy)

			for i in np.range(0, bound):
				it = lm[j]
				beta = np.dot(it.y, d)
				beta = beta / it.ys
				d = d + (it.s * (it.alpha - beta))
				# from former to later
				j = (j + 1) % m
			step = 1.0
		# end while
	#end do_lbfgs