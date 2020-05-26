import numpy as np 
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
import time 

class ADMM_Lasso:
    def __init__(self, X, y, alpha, rho=1.0, rel_par=1.0, MAX_ITER=50, ABSTOL=1e-3, RELTOL= 1e-2):
        self.X = X
        self.y = y
        self.alhpa = alpha 
        self.rho = rho 
        self.rel_par = rel_par
        self.MAX_ITER = MAX_ITER
        self.ABSTOL = ABSTOL
        self.RELTOL = RELTOL

    def objEval(x, z):
        return 0.5 * np.square(self.X.dot(x)-self.y).sum() + self.alpha * norm(z,1)

    def shrinkage(x, kappa):
        return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

    def factor():
        m, n = self.X.shape
        if m >= n:
           L = cholesky(self.X.T.dot(self.X)+self.rho*sparse.eye(n))
        else:
           L = cholesky(sparse.eye(m)+1./self.rho*(self.X.dot(self.X.T)))
        L = sparse.csc_matrix(L)
        U = sparse.csc_matrix(L.T)
        return L,U

    def solveLasso():
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
        m,n = self.X.shape
        # Save a matrix-vector multiply
        Xty = self.X.T.dot(y)

        # ADMM solver
        x = np.zeros((n,1))
        z = np.zeros((n,1))
        u = np.zeros((n,1))

        # Cache the factorization according to the paper
        # We can reuse it
        L, U = factor()

        print ('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

        for k in range(self.MAX_ITER):

            # x-update 
            q = Xty+self.rho*(z-u) #(temporary value)
            if m>=n:
                x = spsolve(U,spsolve(L,q))[...,np.newaxis]
            else:
                ULXq = spsolve(U,spsolve(L,X.dot(q)))[...,np.newaxis]
                x = (q*1./rho)-((self.X.T.dot(ULXq))*1./(self.rho**2))

            # z-update with relaxation
            zold = np.copy(z)
            x_hat = self.rel_par*x+(1.-self.rel_par)*zold
            z = self.shrinkage(x_hat+u,self.alpha*1./rho)

            # u-update
            u += (x_hat-z)

            # diagnostics, reporting, termination checks
            objval   = self.objEval(x,z)
            r_norm   = norm(x-z)
            s_norm   = norm(-self.rho*(z-zold))
            eps_pri  = np.sqrt(n)*ABSTOL+RELTOL*np.maximum(norm(x),norm(-z))
            eps_dual = np.sqrt(n)*ABSTOL+RELTOL*norm(self.rho*u)
            
            print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,r_norm, eps_pri,s_norm,eps_dual, objval))

            if (r_norm < eps_pri and s_norm < eps_dual):
                break

        toc = time.time()-tic
        print("\nElapsed time is %.2f seconds"%toc)

        return z.ravel(),h