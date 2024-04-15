#Importing some useful modules
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import multiprocessing
#Helper Functions
#=================
#Unpads an array after the fourier transform is done
def _unpad2d(array):
    '''Takes a padded array and returns an unpadded array'''   
    #The key here is to realize that it will be 3 times as long as it should be 
    dimension = len(array[0])
    amountToKeep = int(dimension/3)
    exportArray = []
    
    subslice =array[amountToKeep:-1*amountToKeep]
    for i in subslice :
        exportArray.append(i[amountToKeep:-1*amountToKeep])
    
    return np.asarray(exportArray)
#Convolves a wavelet with our image(s) using fourier convolution

def _transformHelper2d(image,kernel,periodic = False):
    ''' 
    Transforms a given image
    Params:
    -------
    image, 2d array: Image being transformed by the wavelet
    kernel,2d array: Wavelet kernel being fit to, must have the same dimensions as imageArray
    periodic, boolean: Whether or not to pad the array, default = False
    '''
    if periodic == True:
        return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(image)*np.fft.rfft2(kernel)))

    #Pad our image
    imagePad = np.pad(image,len(image))
    kernelPad = np.pad(kernel,len(kernel))
    #Convolve
    imagePadFT = np.fft.rfft2(imagePad)
    kernelPadFT = np.fft.rfft2(kernelPad)

    return _unpad2d(np.fft.fftshift(np.fft.irfft2(imagePadFT*kernelPadFT)))

def multiHelper2d(params):
        image,wavelet,periodic = params
        return _transformHelper2d(image,wavelet,periodic)


def autoCorrelation(data):
    ''' Python only implementation '''
    
    length = len(data)
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

    acorr = acorr[0:int(len(acorr)/2)]



    return acorr[:length]

#Wavelet Transform - One Dimensional
#====================================
def waveletTransform1d():
    pass

#Wavelet Transform- Two Dimensional
#====================================
def waveletTransform2d(imageSeries,wavelet,periodic = False):
    ''' 
    Transforms a single image or an image series given choice of wavelet
    Params:
    -------
    imageSeries, 3d array, or 2d array: image(s) to be transformed by the wavelet
    wavelet: Kernel of wavelet to be used to transform our image
    periodic, boolean: Whether or not to pad the input array, default False

    '''
    #Initializing our returned array
    transformedImages = []
   

    #Want to check if it's a single image or not
    if len(np.shape(imageSeries)) == 2:     #If we have a single image
        return (_transformHelper2d(imageSeries,wavelet,periodic))
    
    #This is an ideal candidate for multiprocessing
    #I assume that this is cpu bound because it does a crap* ton of matrix multiplication
    
   
    data = []
    for i in imageSeries:
        data.append((i,wavelet,periodic))
    
    with multiprocessing.Pool(processes =3 ) as pool:
        transformedImages = pool.map(multiHelper2d,data)

    return transformedImages


def correlationWavelet(imageSeries,kernel, periodic =False):
    ''' 
    Returns the wavelet coefficients at the center of the image \n 
    If the image is square (which it will be) the coefficient returned is an \n 
    average if the 4 center squares\n 
    Params:
    -------
    kernel: 2d Array: wavelet used to transform the image
    imageSeries: array[2dArrays]: images we are transforming
    periodic: if the transform is periodic
    Returns:
    --------
    Array[1d]: 1d array with wavelet coefficients
    '''


    #Initialzing required vectors
    returnedArray = []      #Array with our averavge of wavelet coefficients


    #Checking if it is a single image or not
    if len(np.shape(imageSeries)) == 2:
        imageSeries = [imageSeries]

    
    #Define length of the image
    imageSize = np.shape(imageSeries[0])
    #Assume a square image
    center1 = int(imageSize[0]/2 )
    center2 = int(center1-1 )

    for i in imageSeries:

        transform = waveletTransform2d(i, kernel, periodic)
        centers = []
        centers.append(transform[center1,center1])
        centers.append(transform[center1,center2])
        centers.append(transform[center2,center1])
        centers.append(transform[center2,center2])

        returnedArray.append(np.mean(centers))
    
    return returnedArray


def spatialAverage(transform):
    ''' 
    Takes the wavelet transform data of an image series and spatially averages it to return a mean
    and standard deviation of the autocorrelation decay curves
    Params:
    -------
    transform: M,(NXN) Array, a image time series with M steps and NxN pixels

    Returns:
    --------
    mean: the mean of the decay curves, averaged over all the pixels
    std : the standard deviation of the decay curves
    ''' 
    #Preallocating our table 
    transform = np.array(transform)
    dim = np.shape(transform)

    plottable = []
    #Getting the autocorrelation of each pixel
    #Getting the autocorrelation of each pixel
   
    for i in range(dim[1]):
         for j in range(dim[2]):
            plottable.append(autoCorrelation(transform[:,i,j]))

    plottable = np.array(plottable)
    #Getting the means and standard deivaitons of all of our curves
    mean = []
    std = []
    for i in plottable.transpose():
        mean.append(np.mean(i))
        std.append(np.std(i))

    return mean,std

