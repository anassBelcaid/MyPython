from da_denoisers import *
import numpy as np
import shutil
from signalFactory import SignalFactory
def timing_recover(algo,sig,lam=20,h=8,tries=3):
	"""
	function to repeate  and average the timing
	"""

	#temporary save
	S=0
	for i in range(tries):
		(time,Sig)=denoise(algo,sig,lam,h);
		S+=time;
	return S/tries

def compare_fixed_dis(algos,max_size,simulName,samples=30,num_dis=6,tries=3,min_size=128,verbose=False,lam=20,h=8):
	"""
	comparaison of the two algorithms on a fixed number of discontinuities
	"""

	#vector of sizes
	S=np.linspace(min_size,max_size,samples).astype(int)
	num_algos=len(algos)
	R=np.zeros((samples,num_algos+1))
	# recover the mean error of each algorithm
	errors=np.zeros(num_algos)

	#to do remove sigB and sigR
	for (i,s) in enumerate(S):
		print("recovering for size= ",s)
		factory=SignalFactory(s,4) # to be changed
		ini,sig=factory.uniformCase(parts=num_dis);
		#ini,sig=factory.dichotomieCase(parts=num_dis)
		R[i,0]=s;
		for (j,algo) in enumerate(algos):
			R[i,j+1]=timing_recover(algo,sig,lam,h,tries)
			(te,tmp)=denoise(algo,sig,lam,h)
			errors[j]+=mean_square_error(ini,tmp);

	#saving the txt
	np.savetxt(simulName,R,fmt="%.18e",delimiter=',',header=",".join(['S']+algos),comments="")

	#saving the means
	errors/=samples;
	np.savetxt(simulName+"_mse",errors,delimiter=",",header=",".join(algos),comments="")

def compare_fixed_size(algos,max_disc,simulName,samples=30,size=512,tries=3,min_disc=1,verbose=False,lam=20,h=8):
	"""
	comparaison of the two algorithms on a fixed number of discontinuities
	"""

	#vector of sizes
	S=np.linspace(min_disc,max_disc,samples).astype(int)
	num_algos=len(algos)
	R=np.zeros((samples,num_algos+1))
	factory=SignalFactory(size,6) # to be changed
	#to do remove sigB and sigR
	for (i,dis) in enumerate(S):
		print("recovering with",dis," discontinuities for size= ",size)
		#ini,sig=factory.uniformCase(parts=dis);
		ini,sig=factory.normalShapeCase(parts=6)
		R[i,0]=dis;
		for (j,algo) in enumerate(algos):
			R[i,j+1]=timing_recover(algo,sig,lam,h,tries)

	#saving the txt
	np.savetxt(simulName,R,fmt="%.18e",delimiter=',',header=",".join(['S']+algos),comments="")
