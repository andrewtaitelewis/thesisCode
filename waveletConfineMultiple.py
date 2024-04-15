#Import modules
import numpy as np 
import matplotlib.pyplot as plt 
#Modules for simulation practice
import moleculeSimulation as molec 
import waveletAnalysis as WA 
import waveletTransformation as WT 
import scipy
import helper 
#Helper funcitons

#Make our simulation
 #Insert molecules
simulationObject = molec.molecule(numMolec= int(2000))
    #Insert cytoskeleton
numSquares = 16
simulationObject.cytoskeleteonConfinement(numSquares, 0)

#Simulate for some amount of time
imageSeries = simulationObject.simulate(1500, 0.1)
print('here')

#Wavelet Transformation
#======================
    #Make kernel- scale such that wavelet = 0 at edge of skeleton


waveletScales = [10,12,14,16,18,20]
for scale in waveletScales:
    Kernel = WA.KernelCreatorRicker2d(simulationObject, scale )


        #Transform image series with wavelet
    transform = WT.waveletTransform2d(imageSeries, Kernel, periodic= True)
        #Look at coefficient at center

        #Look at coefficient at barrier
    #Variance map
    print(np.shape(np.transpose(transform)))
    transTranspose = np.transpose(transform)

    a = len(transTranspose)
    variance = np.zeros((a,a))


    for i in range(a):
        for j in range(a):
            variance[i][j] = np.std(transTranspose[i][j])**2

    
    
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    for i in simulationObject.xSkeleton:
        ax1.axvline(i*simulationObject.umToPixel)
        ax1.axhline(i*simulationObject.umToPixel)
    

    
    for i in simulationObject.xSkeleton:
        ax1.axvline(i*simulationObject.umToPixel)
        ax1.axhline(i*simulationObject.umToPixel)
    ax1.imshow(Kernel)
    

    for i in simulationObject.xSkeleton:
        ax2.axvline(i*simulationObject.umToPixel)
        ax2.axhline(i*simulationObject.umToPixel)
    

    
    for i in simulationObject.xSkeleton:
        ax2.axvline(i*simulationObject.umToPixel)
        ax2.axhline(i*simulationObject.umToPixel)
    ax2.imshow(variance)
    ax3.imshow(variance)

    #Titles 
    ax2.title.set_text('Variance Map of Image')
    ax2.set(xlabel = '20$um$')
    
    ax1.title.set_text('Wavelet Scale: ' +str(scale))

    plt.savefig('cytoSkeletonConfine'+str(numSquares)+'WaveletScale'+str(scale)+'.png')
    plt.clf()