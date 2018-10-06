"""
Module discontinuity adaptative denoisers

A set of basic functions to test discontinuity adaptative denoisers
"""
from subprocess import check_output
import numpy as np
import ebs
import matplotlib.pyplot as plt

# Dictionary of algorithms binaries
algosBin={"dps":"bpd.exe","gnc":"gnc.exe","rec dps":"rec_bpd.exe","rec dps first": "rec_bpd_first.exe","ebs":"None", "dps learn":"dps_learn.exe"}

def listAlgos():
    """
    return the list of available algos
    """
    return list(algosBin.keys())

def _getAlgobin(algoname):
	"""
    function to return the binary location of an algorithme given it's name
    The list of possible names is stored on the var: listAlgos
  """
	try:
		bin=algosBin[algoname];
		return bin;
	except KeyError:
		print(algoname, " is not a valid denoiser name")
		return None;

def denoise(algoName,sig, lam=20, h=1):
	"""
  Main function to denoise a signal using the algorothm specified by algoName

  algoName: String representing the name (not the binary) of the algorithm
	"""
	np.savetxt("sig_tmp",sig,header="%d"%(len(sig)),comments="");
	if(algoName=="ebs"):
		return ebs.ebs_recover("sig_tmp")

	binName=_getAlgobin(algoName);
	if(binName==None):
		print('no algorithm named : ', algoName)
		return 0

	command=[binName,"-l",str(lam),"-H",str(h),"-i","sig_tmp","-o","out_tmp"]
	#restroing the signal
	time=check_output(command)

	#laoding the signal
	out=np.loadtxt("out_tmp",skiprows=1)

	#return the signal
	try:
		t=float(time)
	except valueError:
		print("output ", time," is not a valid time")
		return None

	return (t,out)

def plot_restoration(ini,noised,denoised):
	"""
	function to plot the restoration case
	"""
	plt.plot(ini,"-",lw=2,label="ini")
	plt.plot(noised,"*",ms=4,lw=0.8,label="noised")
	plt.plot(denoised,"--",lw=2,label="denoised")
	plt.legend(loc="best")
	plt.show()

def mean_square_error(ini,denoised):
	"""
	compute the mean square error
	"""
	n=len(ini)
	sum_squares=np.sum((ini-denoised)**2);

	return np.sqrt(sum_squares)/n;
