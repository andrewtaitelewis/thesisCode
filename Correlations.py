'''
Correlation functions 
''' 
#Importing modules
import numpy as np 

#Functions
#=============================================
#General SpatioTemporal Correlation Function
def SpatialCorrelationFunc(imageA,imageB,xi,eta):
    '''  

    Params:
    -------
    imageA, [(float,float)] 
    imageB, []
    xi 
    eta
    '''
    #Averages of the images
    averageA = np.average(imageA)
    averageB = np.average(imageB)

    #Shifting image B
    #x = 1, y = 0
    #xi is x shift, eta is y shift
    imageB = np.roll(imageB, xi, 1)
    imageB = np.roll(imageB,eta,0)

    #Deltas
    deltaA = imageA-averageA
    deltaB = imageB-averageB

    numerator = np.average(deltaA*deltaB)
    denominator = averageA*averageB
    
    return numerator/denominator

def gaussianFitFunction(X,g0,ginf,w2p):
    xi,eta = X
    ''' 
    Gaussian fit function as seen in herbert et al. number 3
    '''
    return g0*np.exp(-1*(xi**2 + eta**2)/w2p**2) + ginf 

#Correlation Functions 
def temporalCorrelationFunction(imageSeries):
    '''
    Returns the temporal correlation function for a given time series
    Params:
    -------
    imageSeries: the image series to be analyzed
    tau: the number of frames to skip
    '''
    tau = 1
    returnedArray = []      #The returned spatial temporal correlation function

    maximum = len(imageSeries)
    imageIndex = 0 
    
    while tau < maximum:
        imageCorrelations = []
        imageIndex = 0
        while imageIndex + tau < maximum:
            imageA = imageSeries[imageIndex]
            imageB = imageSeries[imageIndex + tau]

            #Now take the averages
            averageA = np.average(imageA)
            averageB = np.average(imageB)
            #The Deltas
            deltaA = imageA - averageA
            deltaB = imageB - averageB

            imageCorrelations.append((np.average(deltaA*deltaB))/(averageA*averageB))
            imageIndex +=1
        returnedArray.append(np.average(imageCorrelations))
        imageCorrelations = []
        tau = tau +1


    return returnedArray

#Decay: Two Dimensional Diffusion
def twoDimensionalDiffusion(x,g0,ginf,td):
    

    return g0*(1+x/td)**(-1) + ginf

   
if __name__ == '__main__' : 
    print('Correlations')

   