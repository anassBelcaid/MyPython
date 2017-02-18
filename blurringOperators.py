"""
module to contain a set of blurring operators
"""
import numpy as np
def gaussian(x,sigma=1):
    """
    function to evaluate the gaussian at x
    """
    return np.exp(-(x**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2);

def gaussian_kernel(l,sigma=1):
    """
    funtion to return the gaussian kernel width lenght l
    """
    X=np.arange(l+1);
    return gaussian(X);

def gaussian_matrix(n,l,sigma=1):
    """
    return the matrix instead of the kennel
    """
    K=gaussian_kernel(l,sigma=sigma);
    mat=np.zeros((n,n));

    for i in range(l):
        if(i==0):
            mat=mat+np.diag(K[i]*np.ones(n-i));
        else:
            mat=mat+np.diag(K[i]*np.ones(n-i),i)+np.diag(K[i]*np.ones(n-i),-i)
    return mat

def mean_matrix(n,l):
    """
    function to return the matrix associate to the moving mean operator
    """

    mat=np.zeros((n,n));
    v=1./(2*l+1);
    for i in range(l):
        if(i==0):
            mat=mat+np.diag(v*np.ones((n)));
        else:
            mat=mat+np.diag(v*np.ones(n-i),i)+np.diag(np.ones(n-i),-i)

    return mat