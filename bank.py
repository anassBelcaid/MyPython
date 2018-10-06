"""
Module to collect a set of signal and propose theme as bank of signals
"""

from signalFactory import SignalFactory
import numpy as np
import matplotlib.pyplot as plt

header="n, rec, equidistant, sampleDec, sample1,sample2"

if __name__ == "__main__":
    
    data=np.loadtxt('signalBank512.csv',delimiter=',')
    n,m= data.shape 

    dataC=np.zeros((n,m+1))

    dataC[:,0]= np.arange(n)
    for i in range(m):
        dataC[:,i+1]=data[:,i]

    np.savetxt('signalBank512n.csv',dataC,delimiter=',',comments="",header =
            header)
    # second size
    data=np.loadtxt('signalBank1024.csv',delimiter=',')
    n,m= data.shape 

    dataC=np.zeros((n,m+1))

    dataC[:,0]= np.arange(n)
    for i in range(m):
        dataC[:,i+1]=data[:,i]

    np.savetxt('signalBank1024n.csv',dataC,delimiter=',',comments="",header =
            header)
