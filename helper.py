import numpy as np 
import matplotlib.pyplot as plt 

#A bunch of helper functions to keep our code as neat as possible
#Visualization
#=============
def timeSeriesVisualizer(timeSeries,pauseTime):


    '''
    Displays a time series of images using plt.imshow()
    Params:
    -------
    timeSeries, array[2dArray]: Images to be shown
    pauseTime, float: Time to pause between images in seconds
    Returns:
    --------
    Nothing
    '''
    plt.ion()
    plt.show()
    
    maxValue = np.max(timeSeries[0])*1.2
    
    for i in timeSeries:
        plt.imshow(i)
        plt.clim(vmin= None, vmax =maxValue)
        plt.colorbar()
        plt.pause(pauseTime)
        plt.clf()
    
    plt.ioff()
    return

#Help plotting microdomains
def circlePlotter(center,radius, ROI,imageResolution, points = 100):
    ''' 
    Plots a circle using 100 points
    Params:
    center, (float,float): center of the circle
    radius, float: radius of the circle
    ROI, area of the region of interest
    imageResolution: area in pixels
    points: number of points:
    '''
    angle = np.linspace(0, 4*np.pi,points)
    SF = imageResolution/ROI    #Our scaling factor
    #Scale everything appropriately
    xCent,yCent = center
    xCent,yCent = xCent*SF,yCent*SF 
    radius = radius*SF

    xs = radius*np.cos(angle) + xCent
    ys = radius*np.sin(angle) + yCent

    return xs,ys

#Lets see how many particles are in the volume at a given time
def numberOfMolecules(simulationObject):
    numberOfMoleculesInImage = []
    molecules = simulationObject
    for i,j in zip(molecules.xPosHist,molecules.yPosHist):
        number = 0
        for y,z in zip(i,j):
            if y > 0 and y<20:
                if z > 0 and z<20:
                    number = number + 1
            
        numberOfMoleculesInImage.append(number)
            
    return np.array(numberOfMoleculesInImage)

#Kernel Creation
#Gaussian
def gaussian(A,xx,yy, sigma ,umToPixel):
    ''' 
    Returns a gaussian centered at 0,0 with a radius in um of 0.4 
    Params:
    -------
    xx,yy: meshgrid of the xx and yy pixels
    sigma: standard deviation of the gaussian in um
    umToPixel: converts um to pixel
    Returns:
    --------
    gaussianArray, float: an array of the gaussian
    '''
    midPoint = max(xx[0])/2
    xx = xx - midPoint
    yy = yy - midPoint
    xx = np.fft.fftshift(xx); yy = np.fft.fftshift(yy)
    
    sigma = umToPixel*sigma
    
    
    return A*np.exp((-2*1/sigma**2)*(((xx-0)**2)+((yy-0)**2)))



#Saving and loading data/models
def saveModel(fName, moleculeObject, diffusionCoefs = [], waveletScales = []):
    ''' 
    Saves the parameters and settings of a given molecule object given the run
    One also needs to save the diffusion coefficient and wavelet Scales used
    Params:
    fname: str: name of file for the model to be saved to
    moleculeObject: object: The molecule object being saved
    diffusionCoefs: [floats]: the diffusion coefficinets used in the simulation
    waveletScales: [Ints]: the wavelet scales used in the wavelet analysis
    '''
    #Importing useful modules
    import json

    a = moleculeObject
    #Store all the stuff in a dictionary
    dictSave ={}        #Intializing our dictionary
    dictSave['DiffusionCoefficients'] = diffusionCoefs
    dictSave['WaveletScales'] = waveletScales
    dictSave['NumMolec'] = a.numMolec
    dictSave['ROI'] = a.ROI
    dictSave['imageResolution'] = a.imageResolution
    dictSave['noiseAmp'] = a.noiseAmp
    dictSave['periodic'] = a.periodic
    dictSave['jumpProb'] = a.jumpProb
    dictSave['skeleton'] = a.skeleton
    dictSave['xSkeleton'] = a.xSkeleton
    dictSave['ySkeleton'] = a.ySkeleton
    dictSave['lipidRaft'] = a.lipidRaft
    dictSave['lipidRaftCenters'] = a.lipidRaftCenters
    dictSave['lipidRaftRadius'] = a.lipidRaftRadius
    dictSave['lipidRaftJumpyProb'] = a.lipidRaftJumpProb
    #Now use pickle to save the stuff
    with open(fName+'.json','w') as f:
        f.write(json.dumps(dictSave))
    return
def loadModel(fName):
    ''' Loads a pickle file to be read
    Params:
    -------
    Filename: the pickle file that contains the information you want
    
    '''
    pass 
def saveData(fName = None ,array = [],dateStamp = True):
    ''' 
    Saves the means/ standard deviation from a spatially averaged data set
    The header of the file will be the original 3d dimensions of the file such as: \n
    # dimensions:(x,y,z)
    '''
    if array == []:
        raise Exception('Array is empty')
    #Check if there is a filename 
    if fName == None:
        fName = '.txt'



    #Get some importing done
    import numpy as np 
    from datetime import datetime
   
    #Get the dimensions and create the header
    dimensions = np.shape(array);header = 'dimensions:'+str(list(dimensions))
    #Now for the filename
    date =  datetime.today().strftime('%Y-%m-%d-%H_%M')
    #FileNames
    if dateStamp == True:
        arrayFname = date+fName 
    else: 
        arrayFname = fName
    #Saving Array
    with open(arrayFname, 'w+') as f:
        #Convert to a 2d Array
        arr_reshaped = array.reshape(array.shape[0],-1)
        np.savetxt(f, arr_reshaped, header= header)
   
    return
def loadData(fName):
    '''
    Loads the means/ standard deviation from a spatially averaged data set
    The header of the file will be the original 3d dimensions of the file such as: \n
    # dimensions:(x,y,z)
    '''
    try:
        with open(fName) as f:
            header = f.readline()
            dimensions = header.split(':')[-1]
            dimensions = dimensions[:-2]
            loadedArray = np.loadtxt(fName)
    except:
        raise Exception('File does not exist or some other error has occured')

    #Change it into something the program can actually read
    res = dimensions.strip('][').split(', ')
    for i in range(len(res)): res[i] = int(res[i])
    dimensions = res
    print(res)
    #Time to read and reshape the file
    print(loadedArray.shape)
    load_originalArray = loadedArray.reshape(loadedArray.shape[0],loadedArray.shape[1] // dimensions[2],dimensions[2])

    return load_originalArray

    

#For testing
if __name__ == '__main__' : 
    print('testing helper')
    xs,ys = circlePlotter((0,0), 1, 1, 1)
    plt.plot(xs,ys)
    plt.show()


