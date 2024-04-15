import numpy as np 
import matplotlib.pyplot as plt 
import moleculeSimulation as molec
import waveletAnalysis as WA 
import waveletTransformation as WT 
import scipy
import helper 
import Correlations
import time 
import os

diffusionCoefficients = [0.1,0.5,1]
numTrials = 20
timeSteps = 100
timeStepSize = 0.02
numMolecules = 5000

#Autocorrelation Code
def autoCorrelation(data):

    ''' Python only implementation '''
    data = correlations
    # Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2*len(data) - 1)).astype('int')

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - np.mean(data)

    # Compute the FFT
    fft = np.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(data)

    acorr = acorr[0:int(len(data))]
    
    return acorr

def model(t,N,td,w):
    return (1/N)*(1+ t/td)**(-1)*(1+t/(w**2 *td))**(-1/2)


#Now for our simulation


#Wavelet Transform
#We will want to try it on a few scales
waveletScales = [1,4,10,16]
transforms = []

for diffusion in diffusionCoefficients:
    wavelets = []
    simulationObject = molec.molecule(numMolec=numMolecules,periodic= True,diffusionCoefficient= diffusion )
    imageSeries = simulationObject.simulate(timeSteps,timeStepSize)
    for scale in waveletScales:
        kernel = WA.KernelCreatorRicker2d(simulationObject,scale)

        #Do the wavelet stuff 
        waveTrans = WT.waveletTransform2d(imageSeries,kernel, periodic= True)
        wavelets.append(waveTrans)
    transforms.append(wavelets)

np.save('transforms',transforms)
        
