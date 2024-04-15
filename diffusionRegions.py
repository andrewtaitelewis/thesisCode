import numpy as np 
import matplotlib.pyplot as plt 
import moleculeSimulation as molec
import matplotlib.pyplot as plt 
import scipy
import helper
import numpy as np
from waveletAnalysis import *
import waveletTransformation as WT
from Correlations import *

imageRez = 128
#Try it out 
simulationObject = molec.molecule(numMolec=100000,diffusionCoefficient= 0.2,imageResolution= imageRez, periodic=True)
simulationObject.diffusionRegion(10,0.7)
timeSteps = 500
timeStepSize = 0.01
imageSeries = simulationObject.simulate(timeSteps,timeStepSize)
#Ok now we should make the 'map' 

waveletScale = 0.3



if __name__ == '__main__':
    
    waveletTransformation = WT.waveletTransform2d(imageSeries,KernelCreatorRicker2d(simulationObject,waveletScale),periodic=True)


    transpose = np.transpose(waveletTransformation)
    res = imageRez
    array =np.zeros((imageRez,imageRez))
    print(np.shape(transpose))
    for i in range(imageRez):
        print(i)
        for j in range(imageRez):
            
                signal = transpose[i][j]
                ogLen = len(signal)
                zeros = np.zeros(len(signal))
                
                signal = np.append(signal,zeros)
                sigFFT = np.fft.fft(signal)
                corr = np.fft.ifft(sigFFT*np.conjugate(sigFFT))[:ogLen]

                
                #Now for some scipy fitting
                ts = np.arange(timeSteps)*timeStepSize
                fitFunc = lambda t,d,a,b: onePop2DRickerModel(t,d,0.4,waveletScale,a,b)
                fit,cov = scipy.optimize.curve_fit(fitFunc,ts,corr/max(corr),bounds= ([0,0,0],[5,np.inf,np.inf]), p0 = [0.5,1e2,0],maxfev =10000)
                
                array[i][j] = fit[0]


    #Now I want to 'blur' it 
    kernel = np.zeros((imageRez,imageRez))
    num = 5 
    for i in range(num):
        for j in range(num):
            x = i - int(num/2)
            y = j - int(num/2)
            kernel[x][y] = 1/num**2

    
    plt.imshow(array)
    plt.colorbar()

    plt.show()
    #Now let's transform it
    kernelFFT = np.fft.fft2(kernel)
    imageFFT = np.fft.fft2(array)
    plt.imshow(np.real(np.fft.ifft2(kernelFFT*imageFFT)))
    plt.colorbar()
    plt.show()
                


