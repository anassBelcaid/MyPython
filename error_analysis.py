# Set of function for estimation error analysis
import numpy as np

def mse(estim, observed):
  """
  compute the mean square error
  """
  nbElements=np.prod(estim.shape)
  return np.sum((estim-observed)**2)/nbElements;


def psnr(estim,observed):

  """
  compute the peak to signal ratio
  """
  M=1;           #value for 8 bits image
  return 20*np.log10(M)-10*np.log10(mse(estim,observed))
