#Testing out single particle tracking code
# 
# =======================================

# Import modules
import numpy as np 
import matplotlib.pyplot as plt 
#Modules for simulation practice
import moleculeSimulation as molec 
import waveletAnalysis as WA 
import waveletTransformation as WT 
import scipy
#Helper funcitons
import time




#Centroid Fitting the wavelet
def particleFinder(image):
    ''' 
    Returns the position of a particle in pixels, single particle only
    Params:
    -------
    image: the image with a single particle
    Returns:
    xPos,yPos: the centroid position in pixels of the particle

    '''
    imageLength = len(image)
    xPosList = []
    yPosList = []
    weights = []
    for i in range(imageLength):
        for j in range(imageLength):
            if image[i][j] <= 0:
                continue 
            else:
                xPosList.append(i*image[i][j])
                yPosList.append(j*image[i][j])
                weights.append(image[i][j])
    xPos = (sum(xPosList)/sum(weights))
    yPos = (sum(yPosList)/sum(weights))

    return xPos,yPos

#New Point Source Function 
def newPSF(pos,a,v,sigma,xcent,ycent):
    xx,yy = pos 
    x = xx - xcent 
    y = yy - ycent
    return 4*a*np.exp(-2*(x**2+y**2)/(v**2+4*sigma**2))*v**2*(v**2 - 2*(x**2+y**2-2*sigma**2))/(np.pi*(v**2 +4*sigma**2)**3)

#Our wavelet
def ricker2d(position,amplitude,centerX,centerY,sigma,a =1,b=1):
    ''' 
    Should pass np.meshgrid to position

    '''
    x,y = position
    
    x = (x-centerX)
    y = (y-centerY)

    zs = amplitude*(1/(4*sigma**4))*(1-0.5*(((x/a)**2 +(y/b)**2)/sigma**2))*(np.exp(-1.0*(((x/a)**2 +(y/b)**2)/(2*sigma**2))))
    return zs/np.max(zs)

#Gaussian
def gaussian(pos,A, sigma ,umToPixel,centerX,centerY):
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
    xx,yy = pos 
    
    
   
    
    sigma = umToPixel*sigma
    
    
    return np.array(A*np.exp((-2*1/sigma**2)*(((xx-centerX)**2)+((yy-centerY)**2)))).ravel()



fitFunc = lambda pos,A,cX,cY: gaussian(pos,A,0.4,256/5,cX,cY)
#Settings
timeSteps = 20
timeStepSize = 0.05
numRuns = 20
#Start at 2:28

confinements =[ 100,250,300,400]
averageGauss = []
averageWave = []
if __name__ == '__main__':
   
    #Initializing data array
    trajectories = []
    trajSTD = []
    trajectories2 = []
    trajSTD2 = []
    for scale in confinements:
        runs = []
        runs2 = []
        for run in range(numRuns):
            print(scale,': ',run)
            #Initializing our object
            simulationObject = molec.molecule(numMolec= 1,periodic = False, S2N = 2,ROI = 5,imageResolution= 256)
            
            #Confinements
            numSquares = int(simulationObject.ROI/(scale/1000))
            
            simulationObject.cytoskeleteonConfinement(numSquares=numSquares,jumpProb= 0)

            #Now making the starting position in the middle of a square
            positionsToStart = simulationObject.xSkeleton
            indexToStart = int(len(positionsToStart)/2)
            startingSpot = (positionsToStart[indexToStart]+positionsToStart[indexToStart+1])/2


            simulationObject.xPositions = np.array([startingSpot]);simulationObject.yPositions = np.array([startingSpot])
        
            startingPixel = startingSpot*256/5
            
            xs = np.arange(simulationObject.imageResolution)
            xx,yy = np.meshgrid(xs,xs)
            #Now for our simulation
            imageSeries = simulationObject.simulate(timeSteps,timeStepSize)

            #Now time to fit our wavelet

            
            #Get the positions 
            xPositions = []; yPositions = []
            waveletScale = 0.2
            fitFunc3 = lambda pos,A,cx,cy: ricker2d((xx,yy),A,cx,cy,waveletScale*256/5).ravel()
            p0 = [1,startingPixel,startingPixel]
            
            timeStart = time.time()
            for i in imageSeries:
                zs = i.copy().ravel()
                zs = zs/np.max(zs)
                fit = [1,1,1]
                #fit,cov = scipy.optimize.curve_fit(fitFunc,(xx,yy),zs,bounds= ([1,0,0.0],[475,256,256]),p0 = p0,maxfev = 800)
                p0 = fit 
                
                x = fit[2]/256*5
                y = fit[1]/256*5
            
                #pixelToUm = simulationObject.ROI/simulationObject.imageResolution

                #x,y = particleFinder(image)     #XPosition in pixels
                xPositions.append(x); yPositions.append(y)
            summation = 0
            for i,j in zip(xPositions,simulationObject.xPosHist):
                summation += abs(i-j)
            print(summation)
            timeEnd = time.time()
            averageGauss.append(timeEnd-timeStart)
            #Mean squared displacement
            xInit = xPositions[0];yInit = yPositions[0]
            MSD = (np.array(xPositions) - xInit)**2 + (np.array(yPositions) - yInit)**2
            #Add it to the MSD
            runs.append(MSD)

            
           
            
            #Now for our second one 
            p0 = [1,startingPixel,startingPixel]
            xPositions = []; yPositions = []
            timeStart = time.time()
            for i in imageSeries:
                zs = i.copy().ravel()
                zs = zs/np.max(zs)
                fit,cov = scipy.optimize.curve_fit(fitFunc3,(xx,yy),zs,bounds= ([1,0,0.0],[475,256,256]),p0 = p0,maxfev = 800)
                p0 = fit 
                
                x = fit[2]/256*5
                y = fit[1]/256*5
            
                #pixelToUm = simulationObject.ROI/simulationObject.imageResolution

                #x,y = particleFinder(image)     #XPosition in pixels
                xPositions.append(x); yPositions.append(y)
            summation = 0
            for i,j in zip(xPositions,simulationObject.xPosHist):
                summation += abs(i-j)
            print('wavelet error: ',summation)
            xInit = xPositions[0];yInit = yPositions[0]
            MSD = (np.array(xPositions) - xInit)**2 + (np.array(yPositions) - yInit)**2
            #Add it to the MSD
            runs2.append(MSD)
            
            
            timeEnd = time.time()

            averageWave.append(timeEnd-timeStart)
            print(np.mean(averageGauss))
            print(np.mean(averageWave))
            
            


        averageTrajectory = []
        averageSTD = []
        for i in np.transpose(runs):
            averageTrajectory.append(np.mean(i))
            averageSTD.append(np.std(i))
        trajectories.append(averageTrajectory)
        trajSTD.append(averageSTD)
        
        averageTrajectory = []
        averageSTD = []
        for i in np.transpose(runs2):
            averageTrajectory.append(np.mean(i))
            averageSTD.append(np.std(i))
        trajectories2.append(averageTrajectory)
        trajSTD2.append(averageSTD)
        
    #Now to save the data to a file
    #np.save('ConfinementsGauss',trajectories)
    #np.save('ConfSTDGauss',trajSTD)
    np.save('ConfinementsWave',trajectories2)
    np.save('ConfinementSTDWave',trajSTD2)

