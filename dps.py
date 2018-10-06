"""Simple module DPS to be extended and cathegorized
"""
import numpy as np
import scipy.linalg as linAlg
import matplotlib.pyplot as plt
from signalFactory import SignalFactory

def from_condensed_to_full(mat):
	"""
	Convert dense representation to full form
	"""

	#getting the size
	n,m=mat.shape

	#constructing the full matrix
	M=np.diag(mat[0,:])
	for k in range(1,n):
		M+=np.diag(mat[k,:m-k],k)+np.diag(mat[k,:m-k],-k)
	return M

def get_dis_positions(rec):
	"""
	simple routine to get the discontinuities position from a vector
	"""
	mask=rec[1:]-rec[:-1]
	return np.nonzero(mask)[0]

def find_nearest_element(Tab,element):
	"""
	function to find the nearest element in an array
	"""
	idx=np.abs(Tab-element).argmin();
	return Tab[idx]

def line_process_loss(I_opt,I_computed):
	"""
	Function to compute the loss funciton as the sum
	\sum (i -i_closest);
	"""

	#computed the closest;
	closest=[find_nearest_element(I_opt,x) for x in I_computed];

	S=0;
	for (i,c) in zip(I_computed,closest):
		S+=int(np.abs(i-c));

	return S;

def Green(lam,a,b,x,xP):
	"""
	Function to convole with a green kernel
	"""
	#constant factor
	f=1/(lam*np.sinh(((b+a)/lam)));
	x_min,x_max=min(x,xP),max(x,xP);

	return f*np.cosh((x_min+a)/lam)*np.cosh((x_max-b)/lam);

def Green2(lam,n,i):
	"""
	Second implementation of Green 2
	"""
	#indice vector
	Idxs=np.arange(n);

	#constant multiplicator
	coef=1/(lam*np.sinh(n/lam));

	#cosh of both vectors
	G=coef*np.cosh(np.minimum(i,Idxs)/lam)*np.cosh((np.maximum(i,Idxs)-n)/lam);
	return G;


def _convolePart(y,lam,i,j):
	"""
	Helper function to convole a given part withe Green kernel
	the concernet part is (i:j) j non inclusive
	"""
	sig=y[i:j];
	n=(j-i);

	rec=np.zeros_like(sig);
	for i in range(n):
		rec[i]=np.sum(sig*Green2(lam,n,i));
	return rec;


def estimate_left(y,lam):
	"""
	estimate the signal a the begining of the segment
	"""
	n=len(y);

	return np.sum(y*Green2(lam,n,0));

def estimate_right(y,lam):
	"""
	estimte at the end of a segment
	"""
	n=len(y);
	return np.sum(y*Green2(lam,n,n-1))

def epsilon_interval(lam,n):
	"""
	the penalty epsilon in a given interval
	"""
	return lam*np.tanh(n/lam);

def energy_increase(y,lam,h):
	"""
	compute the increase in energy in each position
	"""
	alpha=(h*h*lam)/2;
	n=len(y);
	E=np.zeros(n-1);
	for i in range(1,n):
		#getting the signals
		y1,y2=y[:i],y[i:]
		#epsilon
		eps1=epsilon_interval(lam,len(y1));
		eps2=epsilon_interval(lam,len(y2));

		#estimating the signals
		yleft=estimate_left(y2,lam);
		yright=estimate_right(y1,lam);

		#diff
		diff=yleft-yright;

		#proper increase in the energy
		deltaE=diff**2*1/(1/eps1+1/eps2);

		E[i-1]=deltaE-alpha;
	return E;

def restoreByConvolution(y,lp,lam):
	"""
	denoise the signal by retoring each part by a simple convolution with the green
	kernel
	"""
	n=len(y)
	dis=list(sorted(np.nonzero(lp)[0]));
	dis.insert(0,0);
	dis.append(n)
	rec=np.zeros_like(y);

	for i in range(len(dis)-1):
		i,j=dis[i],dis[i+1];
		rec[i:j]=_convolePart(y,lam,i,j)
	return rec;



def mse(y,y_opt):
	"""
	compute the mse error
	"""
	return np.mean(np.sqrt(np.sum((y-y_opt)**2)));

def gemanPriorMatrix(lam,size,factorized=True):
	"""
	generate the banded compressed truncated matrix
	:param self  :
	:param lam   : regularization
	:param size  : size of the matrix
	:return:
	"""
	lam2=lam*lam;
	Ab=np.zeros((2,size));
	Ab[0,0]=1+lam2; Ab[0,size-1]=1+lam2;
	Ab[0,1:-1]=1+2*lam2;
	Ab[1,:]=-lam2;
	if(not factorized):
		return Ab;
	else:
		Ab=linAlg.cholesky_banded(Ab,overwrite_ab=True,lower=True);
		return Ab;

def gemanDiscontMatrix(lam,size,position):
        """
        return the same matrix but considering position as a discontinuity
        :param lam:
        :param size:
        :param position: position of the discontinuity
        :return:
        """
        lam2 = lam * lam;
        Ab = np.zeros((2, size));
        Ab[0, 0] = 1 + lam2;
        Ab[0, size - 1] = 1+lam2;
        Ab[0, 1:-1] = 1 + 2 * lam2;
        Ab[1, :] = -lam2;

        #adding the discontinuity
        Ab[1,position]=0;
        Ab[0,position]-=lam2;
        Ab[0,position+1]-=lam2;

        Ab=linAlg.cholesky_banded(Ab, overwrite_ab=True, lower=True);

        return Ab;

def energy(x,y,lp,lam,h):
	"""
	compute the energy of a configuration x given the observed data y

	The posision of the discontinuities of x are sotred in the line process lp
	the penalizer strenght is lam
	"""
	alpha=h*h*lam/2;
	lam2=lam**2;
	ene=np.linalg.norm(y-x)**2;
	Grad=x[1:]-x[0:-1];
	for (d,l) in zip(Grad,lp):
		if(not l):
			ene+=lam2*d**2;
		else:
			ene+=alpha;

	return ene;


def LpMatrix(lp,lam,factorized=True):
    """
    compute the banded compressed dps matrix for a given line process
    :param lp: numpy array representing the line process
    :param lam: characteristic lenth
    : factorisze: if the function should return the factorized version or not
    """
    lam2=lam*lam;
    size=len(lp)+1;
    Ab=np.zeros((2,size));
    #adding the ones for the  matrix
    Ab[0,:]=1;
    for (i,l) in enumerate(lp):
        if(not l):
            Ab[1,i]-=lam2;
            Ab[0,i]+=lam2;
            Ab[0,i+1]+=lam2;
    if(not factorized):
        return Ab;
    else:
        Ab=linAlg.cholesky_banded(Ab,overwrite_ab=True,lower=True)
        return Ab;

def  denoiseGivenLp(y,lp,lam):
    """
    function to compute the solution given a line process

    :param y: the noised signal:
    :param lp: the fixed line process:
    :param lam: the characteristic lenght
    """
    A=from_condensed_to_full(LpMatrix(lp,lam,False))
    mat=LpMatrix(lp,lam);

    #solving the system
    x,info=linAlg.lapack.dpbtrs(mat,y,lower=True);

    #check
    #print(A.dot(x)-y)
    return x;
