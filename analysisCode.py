import waveletTransformation as WT 
import waveletAnalysis as WA 
import moleculeSimulation as mc
import tifffile
import sys  #We could do argpause instead....
import numpy as np #A classic
import pandas as pd
import scipy
import multiprocessing
import pickle
from numba import jit
#Import File to Analyze 

#We'll just load it from a csv


def model(t,td,D,L,a,w0,sigma):
    '''Catch all hop diffusion model'''
    return a/(((L**2)/3)*(1-np.exp(-t/td))+ 4*D*t + w0**2 +4*sigma**2)**3

fitModel = lambda t,td,D,L,a: model(t,td,D,L,a,0.3,0.4)
def analyze(x):
        imageSeries,timeSteps,whereItGoes = x
        '''What we will run on our multiprocessing'''
        correlation = WA.ticsFFT(imageSeries)[0]    #Correlate
        try:
            fit,cov = scipy.optimize.curve_fit(fitModel,timeSteps,correlation/correlation[0], p0 = [0.02267889,0.05554198,0.09634419,0.38927243], bounds= ([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]))    
        except:
            fit = [-1,-1,-1,-1]#Fit
        return [fit,whereItGoes]

def appender(arrayToAppendTo,Data,window):
    xMin,xMax,yMin,yMax = window
    for i in range(xMin,xMax):
        for j in range(yMin,yMax):
            arrayToAppendTo[i][j].append(Data)
    return

if __name__ == "__main__":
    #Settings
    timeWindowSize = 100       #Number of frames in our time window

    #Load our Analysis Workload
    df = pd.read_csv('Workload.csv')
    numberOfFiles = len(df.index)

    record = df.iloc[0]
    print(record)
    fileName,timeStepSize,ROI = record
    from tifffile import imread
    try:
        imageSeries = imread(fileName)
    except:
        imageSeries = np.load(fileName)
    #
    #Image will have form (N,MxM)
    timeSteps,M,M = np.shape(imageSeries)
    numberOfWindows = timeSteps-timeWindowSize + 1
    numberOfSpatialWindows = M-64+1

    #Initialize an empty list of empty lists of empty lists
    # Dmicro, Ls, Dmacros
    #
    tds = [[[[] for _ in range(M)] for _ in range(M)]for _ in range(numberOfWindows)]; Dmacros = [[[[] for _ in range(M)] for _ in range(M)]for _ in range(numberOfWindows)];Ls = [[[[] for _ in range(M)] for _ in range(M)]for _ in range(numberOfWindows)]


    #1 Wavelet Scales to Analyze on 0.2um, 0.4um , 0.6um

    def makeWork(imageSeries,timeSteps,windowSize):
        '''
        Initializes the tasks for our multiprocessing pool
        Params: 
            Image Series, pre time sliced

        Returns:
            ImageSeries, properly sliced in space
            Where it goes: [xMin,xMax,yMin,yMax,t] where to put it

        '''
        
        workArray = []; imageSeries = np.array(imageSeries)
        for i in range(numberOfSpatialWindows):     #xAxis
            for j in range(numberOfSpatialWindows):                                      #yAxis
                workArray.append([imageSeries[:,i:i+windowSize,j:j+windowSize],timeSteps,[i,i+windowSize,j,j+windowSize]])
        return workArray

                              #Return

    #Size of image

    kernelAid = mc.molecule(ROI=ROI,imageResolution=M)

    #Tranform our images
    wave04 = WA.KernelCreatorRicker2d(kernelAid,0.4)
    wImage04 = WT.waveletTransform2d(imageSeries,wave04)
    wImage04 = np.array(wImage04)

    #Make Work
    ts = np.arange(timeWindowSize)*0.02
    print(numberOfWindows)
    from reprint import output
    for t in range(20): 
        print('Starting: ',t)
        #make work1
        
        toAnalyze = wImage04[t:t+timeWindowSize,:,:]
        workPool = makeWork(toAnalyze,ts,64)
        
        with multiprocessing.Pool(processes= 7) as pool:
            print(len(workPool))
            result = pool.map(analyze,workPool)
            pool.terminate()
        #Append to all the right arrays
            
        print('terminate it')

        for i in result:
            fit,whereItGoes = i
            appender(tds,fit[0],whereItGoes);appender(Dmacros,fit[1],whereItGoes);appender(Ls,fit[2],whereItGoes)
    
    #Save with pickle
    
    with open('tds.pickle', 'wb') as handle:
        pickle.dump(tds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Ds.pickle', 'wb') as handle:
        pickle.dump(Dmacros, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Ls.pickle', 'wb') as handle:
        pickle.dump(Ls, handle, protocol=pickle.HIGHEST_PROTOCOL)
