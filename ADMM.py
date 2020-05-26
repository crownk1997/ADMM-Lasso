import numpy as np 
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
import time 

def objective(X,y,alpha,x,z):
    return .5*np.square(X.dot(x)-y).sum()+alpha*norm(z,1)

def shrinkage(x,kappa):
    return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

def factor(matrix, rho):
    m, n = matrix.shape
    if m >= n:
       L = cholesky(matrix.T.dot(matrix)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(matrix.dot(matrix.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

def lasso_admm(X,y,alpha,rho=1.,rel_par=1.,MAX_ITER=50,ABSTOL=1e-3,RELTOL= 1e-2):
    """
     Solve lasso problem via ADMM
    
     Solves the following problem via ADMM:
    
       minimize 1/2*|| Ax - y ||_2^2 + alpha || x ||_1
    
	 Reference:
     http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """
    # Get the time stamp to output log information
    tic = time.time()

    # Data preprocessing
    m,n = X.shape
    # Save a matrix-vector multiply
    Xty = X.T.dot(y)

    # ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))

    # Cache the factorization according to the paper
    # We can reuse it
    L, U = factor(X,rho)

    print ('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

    for k in range(MAX_ITER):

        # x-update 
        q = Xty+rho*(z-u) #(temporary value)
        if m>=n:
            x = spsolve(U,spsolve(L,q))[...,np.newaxis]
        else:
            ULXq = spsolve(U,spsolve(L,X.dot(q)))[...,np.newaxis]
            x = (q*1./rho)-((X.T.dot(ULXq))*1./(rho**2))

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = rel_par*x+(1.-rel_par)*zold
        z = shrinkage(x_hat+u,alpha*1./rho)

        # u-update
        u+=(x_hat-z)

        # diagnostics, reporting, termination checks
        objval   = objective(X,y,alpha,x,z)
        r_norm   = norm(x-z)
        s_norm   = norm(-rho*(z-zold))
        eps_pri  = np.sqrt(n)*ABSTOL+RELTOL*np.maximum(norm(x),norm(-z))
        eps_dual = np.sqrt(n)*ABSTOL+RELTOL*norm(rho*u)
        
        print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,r_norm, eps_pri,s_norm,eps_dual, objval))

        if (h['r_norm'][k]<h['eps_pri'][k]) and (h['s_norm'][k]<h['eps_dual'][k]):
            break

    toc = time.time()-tic
    print("\nElapsed time is %.2f seconds"%toc)

    return z.ravel(),h