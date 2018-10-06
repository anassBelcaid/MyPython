"""
subroutines for ebs solver
"""
import numpy as np
from time import time
from subprocess import check_output

## recovering with ebs
def find_lambda(inputName):
    command=['lambdaopt','--input',inputName]
    val=check_output(command)
    return float(val)

def condat_denoise(lam,inputName="noised",outputName="denoised"):
    """
    denoise the signal given the regularization parameter
    """
    command=['denoising','--lambda',str(lam),'--input',inputName,'--output',outputName]
    val=check_output(command)

def cluster_signal(clusterName="denoised"):
    distance = 0.34
    cmd = ['level_generator',
           '--level-distance', str(distance),
           '--input', clusterName,
           '--output', 'level_data'];
    check_output(cmd)

    cmd = ['graph_processing',
           '--levels', 'level_data',
           '--rho-d', str(0.4),
           '--rho-s',  str(0.2),
           '--rho-p',  str(0.02),
           '--prior-distance', str(distance),
           '--input', clusterName,
           '--output', 'clustered_data'];
    check_output(cmd)

def ebs_recover(inputName="noised"):
    """
    function to reconstruct the signal
    """
    t=time();
    lam=find_lambda(inputName)
    condat_denoise(lam,inputName)
    cluster_signal()
    t=time()-t;
    rec=np.loadtxt('clustered_data',skiprows=2)
    return (t*1000,rec);
