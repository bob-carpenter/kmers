import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

## Plot 2-d version of l_i(\alpha) =  \log \left( x_i^{\top} \cdot \sigma(\alpha) \right)
size=2
xs = np.random.rand(size)
xs = xs/xs.sum()

xs = np.random.rand(size)
xs = xs/xs.sum()
x1 = np.arange(-10, 10, 0.01)
x2 = x1
x1, x2 = np.meshgrid(x1, x2)
  
fig = plt.figure()
axes = fig.gca(projection ='3d')
axes.plot_surface(x1, x2, -np.log(xs[0] *np.exp(x1)/(np.exp(x1)+np.exp(x2))  +xs[1]* np.exp(x2)/(np.exp(x1)+np.exp(x2))) )
axes.set_xlabel(r"$\alpha_1$", fontsize=15, rotation=60)
axes.set_ylabel(r"$\alpha_2$", fontsize=15, rotation=60)
axes.set_zlabel(r"$\ell$", fontsize=15, rotation=60)
plt.show()


## Simple non-convex example for paper
size =2
X = [ [0.25, 0.5], [0.75 ,0.5]]
xs = X[1,:]
a1 = np.linspace(-5, 5, 50)
l = lambda a: -np.sum(np.log(X@ np.exp(a)/(np.sum(np.exp(a))))) +a@a/(2*100*100)
# l = lambda a: np.log(xs@ np.exp(a)/(np.sum(np.exp(a))))
fslice = []
a = np.zeros(2)
for x in a1:
      a[0] =x
      a[1] =-x
      fslice.append(l(a)) 

plt.title(r"Plot of $\ell(\alpha)$ with $\alpha_1$ =$-\alpha_2$", fontsize=20) 
plt.ylabel(r"$\ell(\alpha)$", fontsize=20) 
plt.xlabel(r"$\alpha$", fontsize=20) 
plt.plot(fslice, linewidth=5) 
plt.show()


## Contour 2-d plot of \ell_i
def ll2(x1, x2):
    return -np.log(xs[0] *np.exp(x1)/(np.exp(x1)+np.exp(x2))  +xs[1]* np.exp(x2)/(np.exp(x1)+np.exp(x2))) 

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = ll2(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100) #, cmap='binary'
ax.set_xlabel(r'$\alpha_1$')
ax.set_ylabel(r'$\alpha_2$')
ax.set_zlabel(r'$\ell$');
plt.show()

## Testing numericall convexity if ell(x) - ell(y) - 2*ell((x+y)/2) >= 0
size=100

X = np.random.rand(size, size)
X = X/X.sum(axis =0)

xs = X[1,:]
# xs = np.random.rand(size)
# xs = xs/xs.sum()
l = lambda a: -np.sum(np.log(X@ np.exp(a)/(np.sum(np.exp(a))))) +a@a*0.001
l0 = lambda a: -np.log((X@ np.exp(a)+1)/(np.sum(np.exp(a)) +1)) # With bijective softmax
testnum =100
a1 = np.random.randn(size,size)*10
a1 = a1 -np.mean(a1)
a2 = np.random.rand(size, size)*10
a2 = a2 -np.mean(a2)
conv_test = []
ftest = l
for x in a1:
      for y in a2:
            # print(" x, y, l(x), l(y) = ", x, y, l(x), l(y))
            conv_test.append(ftest(x) +ftest(y) - 2*ftest((x+y)/2)) 

plt.title("Convexity test") 
plt.plot(np.sort(conv_test)) 
plt.show()



## Testing numericall quasi-convexity if ell(lam* x + (1-lam)*y) <= max (ell(x), ell(y))
size=100
# xs = np.random.rand(size)
# xs = xs/xs.sum()
l = lambda a: -np.log(xs@ np.exp(a)/(np.sum(np.exp(a))))
testnum =100
a1 = np.random.rand(size,size)*10
a1 = a1 -np.mean(a1)
a2 = np.random.rand(size, size)*10
a2 = a2 -np.mean(a2)
conv_test = []
ftest = l
for x in a1:
      for y in a2:
            lamb = np.random.rand(1)
            # print(" x, y, l(x), l(y) = ", x, y, l(x), l(y))
            conv_test.append(np.maximum(ftest(x), ftest(y)) -  ftest( lamb*x+(1-lamb)*y)) 

plt.title("Convexity test") 
plt.plot(np.sort(conv_test)) 
plt.show()

## Test the full Hessian before nabla sig substitution

 #\nabla li = - \frac{\nabla \sigma(\alpha)  \diag{ x_i}}{ x_i^{\top} \cdot \sigma(\alpha) } 
 # + \frac{ (\nabla \sigma(\alpha) x_i)(\sigma(\alpha)  \odot x_i)^\top}{( x_i^{\top} \cdot \sigma(\alpha) )^2}  
  #+\nabla \sigma(\alpha)

size=1000
sigs = np.random.rand(size)
sigs = sigs/sigs.sum()
xs = np.random.rand(size)
xs = xs/xs.sum()
xsTsigs  = xs@sigs
Dsigs = np.diag(sigs) - np.outer(sigs,sigs)
Hessli = -Dsigs@np.diag(xs)/xsTsigs + np.outer(Dsigs@xs, sigs*xs)/(xsTsigs**2) + Dsigs
eigs, vs = np.linalg.eig(Hessli)



## Testing reduced hessian form
v = xs*sigs/(xs@sigs)

Hess_v = np.outer(v,v) - np.diag(v) + Dsigs
eigv, vv = np.linalg.eig(Hess_v)

Hessdiff = np.linalg.norm(Hess_v- Hessli)/np.linalg.norm(Hessli)
print("Hessdiff = ", Hessdiff)
plt.title("Eigenvalues") 
plt.xlabel("") 
plt.ylabel("sorted eigenvalues") 
plt.plot(np.sort(eigs)) 
plt.show()


## Looking at Hessian of sigs*x to see if it's concave
#  \diag{ x \circ \sigma} -\diag{\sigma}(\sigma(\alpha) \circ x )
#  - \big(\sigma(\alpha) (\sigma(\alpha)\circ x) + (\sigma(\alpha) \circ x) \sigma(\alpha)^\top\big)
#  +2 \sigma(\alpha) \sigma(\alpha)^\top (\sigma ^\top x) 



Hess = np.diag(sigs*xs) -np.diag(sigs)*(sigs@xs) - (np.outer(sigs,sigs*xs) + np.outer(sigs*xs, sigs) ) + 2*np.outer(sigs,sigs)*(sigs@xs)
# eigs, vs = np.linalg.eig(Hess)
