#Importing useful modules 
import numpy as np 
import matplotlib.pyplot as plt 

'''  
Generating Wavelets

'''
#Fit Functions:
#=========================================================================================================
def onePop2DRickerModel(t,D,v,sigma,a,b):
    '''
    The fit function for one population two dimensional diffusion transformed with a ricker wavelet: \n
    Fit this function to the tics curve generated 
    Params:
    -------
    t, array[floats]: tau(s) in seconds
    D, float: diffusion coefficient um^2/s
    v, float: psf e^-2 radius, in um
    sigma, float: standard deviation of the wavelet used to transform the image, um
    a: Zero lags parameter/ amplitude
    b: Infinite Lag Amplitude
    Returns:
    G(0): Value of the correlation function given the parameters
    '''
    d = D
    return a/((4*d*t + v**2 +4*sigma**2)**3) + b

    #a*v**4/((4*d*t + v**2 +4*sigma**2)**3) + b
    #I substituded 3 for 1

#=========================================================================================================
#Temporal Image Correlation Function
def ticsFFT(imageSeries, meanDivision = False):
    '''
    Returns the temporal image correlation function of the image series
    Assuming a 3d Array of (imageTimeIndex, xAxis,yAxis)
    Params:
    -------
    Transformed Image Series
    Mean = False, when false will not normalize by mean, this is because wavelet images have a mean of 0
    Returns:
    --------
    Correlation Function, array[floats]
    Std, array[floats]: Standard deviation of correlation function
     '''
    #Turns our image series into a fluctuation
    preprocess = []
    #If we are dividing by our mean
    if meanDivision == True:
        for i in imageSeries:
            tempMean = np.mean(i)
            preprocess.append(((i - tempMean))/tempMean) 
    else:
        for i in imageSeries:
             preprocess.append((i - np.mean(i))) 
    #Pads our image series with 0s to avoid the circular convolution
    padding = np.zeros(np.shape(imageSeries[0]))
    for i in range(len(imageSeries)-1):
            preprocess.append(padding.copy())
    
    OGlen = len(imageSeries)        #So we know how to get our image back
    #Fourier Magic, axis 0 to make sure we're fourier-ing in time
    convFFT = np.fft.fft(preprocess,axis = 0)
    signal = np.fft.ifft(convFFT*np.conjugate(convFFT),axis = 0)

    signal = signal[:OGlen]     #Get back our signal

    newSig = []
    index = OGlen
    std = []
    signal = np.real(signal)
    #Make sure it is the right amplitude
 
    for i in signal:

        newSig.append(np.mean(i)/index)
        std.append(np.std(i)/index)
        index = index - 1
    return newSig,std

def crossCorrFFT(imageSeries1,imageSeries2, meanDivision = False):
    '''
    Returns the temporal image correlation function of the image series
    Assuming a 3d Array of (imageTimeIndex, xAxis,yAxis)
    Params:
    -------
    Transformed Image Series
    Mean = False, when false will not normalize by mean, this is because wavelet images have a mean of 0
    Returns:
    --------
    Correlation Function, array[floats]
    Std, array[floats]: Standard deviation of correlation function
     '''
    #Turns our image series into a fluctuation
    preprocess = []
    preprocess2 = []
    #If we are dividing by our mean
    if meanDivision == True:
        for i in imageSeries1:
            tempMean = np.mean(i)
            preprocess.append(((i - tempMean))/tempMean) 

        for i in imageSeries2:
            tempMean = np.mean(i)
            preprocess2.append(((i - tempMean))/tempMean) 
    else:
        for i in imageSeries1:
            preprocess.append((i - np.mean(i))) 
        for i in imageSeries2:
            preprocess2.append((i - np.mean(i))) 

    #Pads our image series with 0s to avoid the circular convolution
    padding = np.zeros(np.shape(imageSeries1[0]))
    for i in range(len(imageSeries1)-1):
            preprocess.append(padding.copy())
            preprocess2.append(padding.copy())
    
    OGlen = len(imageSeries1)        #So we know how to get our image back
    #Fourier Magic, axis 0 to make sure we're fourier-ing in time
    convFFT = np.fft.fft(preprocess,axis = 0)
    conv2FFT = np.fft.fft(preprocess2,axis = 0)
    signal = np.fft.ifft(convFFT*np.conjugate(conv2FFT),axis = 0)

    signal = signal[:OGlen]     #Get back our signal

    newSig = []
    index = OGlen
    std = []
    signal = np.real(signal)
    #Make sure it is the right amplitude
 
    for i in signal:

        newSig.append(np.mean(i)/index)
        std.append(np.std(i)/index)
        index = index - 1
    return newSig,std



#Ricker Wavelets
#=========================================================================================================
#One Dimensional
def ricker1d(t,center,sigma):
    t = t-center
    return (2.0/(np.sqrt(3*sigma)*np.pi**(1/4))*(1- (t/sigma)**2)*np.exp(-1.0*(t**2)/(2*sigma**2)))

#Two Dimensional
def ricker2d(position,center,sigma,a =1,b=1):
    ''' 
    Should pass np.meshgrid to position

    '''
    x,y = position
    centerX,centerY = center 
    x = (x-centerX)
    y = (y-centerY)

    return (1/(np.pi*sigma**4))*(1-0.5*(((x/a)**2 +(y/b)**2)/sigma**2))*(np.exp(-1.0*(((x/a)**2 +(y/b)**2)/(2*sigma**2))))

def KernelCreatorRicker2d(simulationObject,sigma,a=1,b=1,theta = 0):
    ''' 
    Returns a 2dRicker wavelet given a given simulation object 
    Params:
    -------
    SimulationObject, object: an object of the moleculeSimulation class
    Sigma, float: the sigma parameter in the wavelet in um 
    a, float: the elongation of the wavelet 
    Returns:
    --------
    Kernel, 2d Array_like, float: the kernel of the wavelet we have created
    '''
    resolution = simulationObject.imageResolution
    sigma = sigma*simulationObject.umToPixel        #Changing the um to pixels

    xs = np.arange(resolution)

    midPoint = max(xs)/2
    xs = xs - midPoint
    
    xs = np.fft.fftshift(xs)
    xx,yy = np.meshgrid(xs,xs)
    kernel = ricker2d((xx,yy), (0,0), sigma,a,b)
    

    return np.fft.fftshift(kernel) 

#=========================================================================================================
#Morelet Wavelets
#=========================================================================================================








