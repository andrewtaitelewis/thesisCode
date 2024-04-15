#Simulations to deterine the cytoskeletal simulations
# 
# 
# Importing useful modules
import numpy as np 
import matplotlib.pyplot as plt 
import moleculeSimulation as molec 
import helper
import waveletAnalysis as WA 
import waveletTransformation as WT
import scipy
import numpy 
import time
import os
#Helper Functions
def exponentialDecay(x,a,b,c):
    return a*np.exp(-b*x) + c





#Create/ Seed the settings for the simulation + analysies
#=======================================================
diffusionCoefs = np.arange(10)*0.1 + 0.1
waveletScales = [1,2,4,8,16,32,64]
#Simulation Parameters 
numMolecules = 3000
timeSteps = 500
interval = 0.02 #In seconds
numTrials = 5 #Trials per setting


#==============================
#Run the different analysis and such

#Preseed our results, rows = diffusionCoef, columns = waveletscale
results = np.zeros((len(diffusionCoefs),len(waveletScales),numTrials))      #Our exponential decay curves
resultsSTD = results.copy()                                                 #STD of our exponential decay curves


#timer
trialsCompleted = 0
startTime = time.time()
totalNumberOfSimulations = len(waveletScales)*len(diffusionCoefs)*numTrials

for coefIndex in range(len(diffusionCoefs)):
    coefficient = diffusionCoefs[coefIndex]
    
   
    for trial in range(numTrials):

    #Timer =========
        timeNow = time.time()
        elapsedTime = timeNow -startTime
        if trialsCompleted != 0:
            
            print('Diffusion: ', coefficient)
            print('-----------------------')
   
            averageTime = float(elapsedTime/trialsCompleted )
            remainingTime = averageTime*(totalNumberOfSimulations - trialsCompleted)
            elapsedTime = remainingTime
            elapsedHours = (elapsedTime)//3600
            elapsedMinutes = (elapsedTime%3600)//60
            elapsedSeconds = (elapsedTime%60)
            print('Time Remaining: ', elapsedHours, ' Hours, ', elapsedMinutes, 'Minutes, and ', elapsedSeconds, 'Seconds (approximately)')
        #TImer Ends ======
        
        #Run the simulation
        simulationObject = molec.molecule(numMolec= numMolecules,diffusionCoefficient= coefficient, periodic= True, noise = 0, ROI= 100)
        simulationObject.cytoskeleteonConfinement(3, 0)
        imageSeries = simulationObject.simulate(timeSteps= timeSteps, timeStepSize= interval)
        for waveletIndex in range(len(waveletScales)):
            
            scale = waveletScales[waveletIndex]

            
            kernel = WA.KernelCreatorRicker2d(simulationObject, scale)      #Our wavelet
            transform = WT.waveletTransform2d(imageSeries, kernel,periodic= True)          #Transforming our image
            mean,std = WT.spatialAverage(transform)                         #The spatial transformation of our wavelet
            
            #now we will fit the curve
            fit = [0,0,0]
            
            while fit[1] == 0:
                try:
                    
                    xs = np.arange(timeSteps)*interval
                    initialGuess = np.random.rand()*2
                    fit,pcov = scipy.optimize.curve_fit(exponentialDecay,xs,mean,p0 =[1,initialGuess,1],sigma= std)
                    
                except:
                    continue
            #After we finally fit our results
            results[coefIndex][waveletIndex][trial] = fit[1]
            
                
            resultsSTD[coefIndex][waveletIndex][trial] = np.sqrt(pcov[1][1])
            trialsCompleted += 1
print('Finished')

#Save our data
#Saving means 
helper.saveData('cytoskeletalMean2.txt',results,dateStamp=False)
#Saving STD
helper.saveData('cytoskeletalSTD2.txt',resultsSTD,dateStamp= False)



