import waveletTransformation as WT 
import waveletAnalysis as WA 
import moleculeSimulation as mc
import tifffile
import sys  #We could do argpause instead....
import numpy as np #A classic
import pandas as pd
import scipy
import multiprocessing
import os; import shutil
import time
import zarr

#See if the data directory exists and if so delete it
if(os.path.isdir('data')):
    shutil.rmtree('data')
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
    if M > 512:
        raise Exception("Image too large, the maximum size is 512x512")
    numberOfWindows = timeSteps-timeWindowSize + 1
    numberOfSpatialWindows = M-64+1

    #Preallocated Numpy arrays for Dmicro, Ls, Dmacros
    print('Allocating memory for coefficient arrays')
    tds = np.full((64**2,M,M), np.nan )
    Dmacro = np.full((64**2,M,M), np.nan )
    Ls = np.full((64**2,M,M), np.nan )
    print('Arrays allocated')


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
    print('Transforming image...')
    wave04 = WA.KernelCreatorRicker2d(kernelAid,0.4)
    wImage04 = WT.waveletTransform2d(imageSeries,wave04)
    wImage04 = np.array(wImage04)

    #Make Work
    ts = np.arange(timeWindowSize)*timeStepSize
    
    print('Starting analysis...')

    #Initializing zarr file
    tdsZarr = zarr.create_array(store="data/tds.zarr",shape=(1000, 64**2,M,M),chunks=(1,64**2,M, M),dtype="f8")
    dMacrosZarr = zarr.create_array(store="data/dMacros.zarr",shape=(1000, 64**2,M,M),chunks=(1,64**2,M, M),dtype="f8")
    LsZarr = dMacrosZarr = zarr.create_array(store="data/Ls.zarr",shape=(1000, 64**2,M,M),chunks=(1,64**2,M, M),dtype="f8")
    for t in range(len(imageSeries)): 
        print('Starting run:',t)
        #make work1
        time1 = time.time()
        toAnalyze = wImage04[t:t+timeWindowSize,:,:]
        workPool = makeWork(toAnalyze,ts,64)
        
        print('Making Work took %s seconds' % str((time.time() - time1))[:6])
        time1 = time.time()
        #
        
        with multiprocessing.Pool(processes= 7) as pool:
            print(len(workPool))
            result = pool.map(analyze,workPool)
            pool.terminate()

            print('Multiprocessing took %s seconds' % str((time.time() - time1))[:6])
        #Append to all the right arrays
            
       

        print('appending results')
        time1 = time.time()


        indexer = np.zeros((M,M),dtype='int')
        for i in result:
            
            fit,whereItGoes = i
            xMin,xMax,yMin,yMax = whereItGoes

            for xs in range(xMin,xMax):
                for ys in range(yMin,yMax):
                    

                    tds[indexer[xs,ys],xs,ys] = fit[0]
                    Dmacro[indexer[xs,ys],xs,ys] = fit[1]
                    Ls[indexer[xs,ys],xs,ys] = fit[2]
                    indexer[xs,ys] += 1


           
        print(np.max(indexer)) 
        print('Appending took %s seconds' % str((time.time() - time1))[:6])
    
        #Write to file
        time1 = time.time()
        tdsZarr[t] = tds
        dMacrosZarr[t] = Dmacro
        LsZarr[t] = Ls
        print('Writing to file took %s seconds' % str((time.time() - time1))[:6])