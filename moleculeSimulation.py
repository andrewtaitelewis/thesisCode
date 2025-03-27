#Method that holds the code for the molecule simulations
# 
# Importing useful modules
import numpy as np 
import matplotlib.pyplot as plt
import helper
import scipy
import os 
import PIL 
import gillespie

#The class that contains our cell simulation

np.random.seed(1)
class line:
    def __init__(self,x1,y1,x2,y2,p = 0):
        '''
        Defines our line
        '''
        self.x1 = x1; self.y1 = y1  #Coordinates of one end point of the line
        self.x2 = x2; self.y2 = y2  #Coordinates of the second end point of the line
        self.p = p                  #Probability for a molecule to jump over the line
        pass
    
    
    def returnPoints(self):
        '''Returns the points which define the line'''
        return self.x1,self.y1,self.x2,self.y2


def makeLine(array):
    '''
        Makes a line object
    '''
    newLine = line(*array)
    return newLine

class molecule:
    '''
    Object that represents a bunch of molecules in a volume
    Params:
    -----------
    numberOfmolecules; int; number of molecules that we will create
    diffusionCoefficient: float: the diffusion coefficient in um^2/s of the molecules 
    ROI, 20: region in um per side of the image
    noise, 0.5: Amplitude of the gaussian white noise
    S2N, 100: the signal to noise ratio of the peaks vs. the noise
    periodic, False, If the molecules wrap around or not
    Returns:
    ----------
    molecule Object: Used to simulate the diffusion or flow of a population of particles
    '''
    
    def __init__(self,numMolec = 30, diffusionCoefficient = 0.1, xVelocity = 0, ROI = 20,imageResolution = 256, 
    S2N =100,periodic = True):
        #Setting our coefficients
        self.numMolec = numMolec
        self.diffusionCoefficient = diffusionCoefficient
        self.ROI = ROI                              #Region of interest in um
        self.imageResolution = imageResolution      #Number of pixels in our image
        self.S2N = S2N
        self.periodic = periodic

        self.xVelocity = xVelocity

        #Cytoskeleton
        self.jumpProb = 1           #Probability a walker can jump
        self.skeleton = False       #Is there a skeleton there
        self.xSkeleton = []         #X coordinates of the skeleton
        self.ySkeleton = []         #Y coordinates of the skeleton

        #Lipidrafts
        self.lipidRaft = False      #Do we have lipid rafts
        self.lipidRaftCenters = []
        self.lipidRaftRadius = []
        self.lipidRaftJumpProb = []

        #Blinking
        #Blinking for two population
        #We use gillespie to model the transition 
        self.blinking = False
        self.rateOn = 0             #Transition from state A to B
        self.rateOff = 0             #Transition from state B to A
        self.onMolecules = np.ones(self.numMolec)       #An array specifiying the 'On' molecules =1 
        self.numOn = numMolec
        
        
        self.floureHist = []        #The amount on at a time FOR TESTING VALIDITY


        #Now for cell positions
        self.xPositions = np.random.rand(numMolec)*ROI
        self.yPositions = np.random.rand(numMolec)*ROI 
        
        #History of cell positions
        self.xPosHist = []
        self.yPosHist =[] 

        #Other 
        self.umToPixel = self.imageResolution/self.ROI
        self.beamRadius = 0.4      #point source function in um, where values fall to 33 percentish, i.e 1 sigma
        
        #Two diffusion
        self.twoDiffusion = False



        self.confinements = []      #The lines we create

#Cytoskeleteon confinement
    def confinementInitializer(self, array):
        '''
        sets up the confinement given an array of confinements
        Array = [N,5] (x1,y1,x2,y2,p)
        '''
        for i in array:
            self.confinements.append(makeLine(i))

        self.skeleton = True        #Makes the simulator aware that we have confinements
        return

    def cytoskeleteonConfinement(self, numSquares,jumpProb):
        ''' 
        cytoskeletalConfinement: creates a meshwork skeleton for the particles to diffuse within
        ------------------------
        numSquares, int: number of squares across the ROI, i.e. 3 squares would mean 9 total
        jumpProb, float, 0 < 1: the probability that if a walker tries to jump across the cytoskeleton
        it will succeed
        '''
        #Update our internal variables
        self.jumpProb = jumpProb
        self.skeleton = True
        #Now time to make the skeleton
        #Stretches for 2x roi in each direction
        lineLocations = np.linspace(0-2*self.ROI,self.ROI+2*self.ROI,numSquares*5+1)

        self.xSkeleton = lineLocations
        self.ySkeleton = lineLocations.copy()

        return

    #LipidDomain Confinement
    def lipidDomainConfinement(self, radius,location,jumpProbability = 0.1):
        ''' 
        Creates circular lipid domains for confinement with their own diffusion coefficients
        Params:
        -------
        radius, float: size of microdomain in um
        location, (float,float): where the microdomain is, in um
        jumpProbabilities, float, 0,1: the probability that a particle will jump over a membrane
        '''
        self.lipidRaftCenters.append(location)
        self.lipidRaftRadius.append(radius)
        self.lipidRaft = True 
        self.lipidRaftJumpProb.append(jumpProbability)
        return

    #Export our 'image'
    def imageExport(self):
        ''' 
        Export an image of the molecules, \n
        seed all of the molecules onto a 2048x2048 grid and then bin them down to the image resolution
        Params:
        -------
        Returns:
        --------
        '''
        #Helper function 
        def rebin(arr, new_shape):
            shape = (new_shape[0], arr.shape[0] // new_shape[0],
                    new_shape[1], arr.shape[1] // new_shape[1])
            return arr.reshape(shape).mean(-1).mean(1)

        #Bringing some stuff from the class instance
        imageResolution = self.imageResolution
        subImageRes = imageResolution*4
        #Load the gaussian kernel 
        #========================
        x = np.arange(subImageRes)
        xx,yy = np.meshgrid(x,x)
        kernel = helper.gaussian(1,xx, yy, self.beamRadius,subImageRes/self.ROI)      #Gaussian Beam


        
        
        image = np.zeros((subImageRes,subImageRes))       #our pixels

        
        
        
        #Change it so that it just resolves 9t 
        #now we gotta place our molecule in the x,y field 
        xIntegerPositions = (np.round(self.xPositions*(float(subImageRes/self.ROI))))
        yIntegerPositions = (np.round(self.yPositions*(float(subImageRes/self.ROI))))
        
        for i,j,k in zip(xIntegerPositions,yIntegerPositions,self.onMolecules):
            #Make sure our molecule is actually in our image
            if i < 0 or i > subImageRes-1 or j < 0 or j > subImageRes - 1:
                continue 
            #If blinking and it is off, we will continue
            if self.blinking and k == 0:
                continue



            image[int(i)][int(j)] = 1      #i.e. we have a molecule there

        

        
        #Now we want to convolve it with a gaussian
        imageFT = np.fft.rfft2(image)
        gaussianFT= np.fft.rfft2(kernel)

        returnedImage = np.fft.irfft2(imageFT*gaussianFT) 
        #Now we will want to bin it to our image resolution
        returnedImage = rebin(returnedImage,(self.imageResolution,self.imageResolution))
        
        
        
        return returnedImage

    def gSim(self,rateOn,rateOff,maxTime):
            ''' 
            Initializes our gillespie simulation and lets it burn into steady state
            Params:
            Self: the simulation object
            rateOn, float: the rate that molecules go OFF -> ON
            rateOff, flaot: the rate that molecule go ON -> OFF
            Returns:
            Null
            '''
            self.blinking = True        #Indicate that we have blinking in our system
            #Initialize our variable
            self.rateOn = rateOn 
            self.rateOff = rateOff

            #We will use the gillespie package
            if self.numMolec%2 != 0:        #If our number of molecules is odd (for whatever reason)
                initial = [int(self.numMolec+1)/2,int(self.numMolec+1)/2 - 1]
            else:
                initial = [int(self.numMolec+1)/2,int(self.numMolec+1)/2 - 1]

            #[Number On Molecules, Number Off Molecules]

            #Our propensities and update
            propen = [lambda s,r: s*rateOff, lambda s,r: r*rateOn]
            rates = [[-1,1],[1,-1]]

            #Now let it burn in
            times,measurements = gillespie.simulate(initial,propen,rates,duration = 100)
            #=== Post burn in we can now run it at steady state ===# 
            init = measurements[-1]
            times,measurements= gillespie.simulate(init,propen,rates,duration = maxTime)

            return measurements,times
   
    #Diffuse our molecules
    def diffuse(self,timeStepSize):
        ''' 
        diffuses the cells based on a random diffusion process based on fick diffusion
        Params:
        -------
        timeStepSize: how large of a time step is taken (s)
        ''' 
            
        def positionGenerator(diffusionCoefficient,timeStepSize,numSamples):
            '''
            Diffusing
            '''
            d = diffusionCoefficient
            t = timeStepSize
            return 2*np.sqrt(d*t)*scipy.special.erfinv(2*(np.random.uniform(size = numSamples)-0.5))
        
        
        #Generating our x jumps
        xOffset = (positionGenerator(self.diffusionCoefficient, timeStepSize, self.numMolec))
        if self.xVelocity != 0:
            jump = self.xVelocity*timeStepSize
            xOffset += jump
        
        
        xOffset = np.asarray(xOffset)
        yOffset = (positionGenerator(self.diffusionCoefficient, timeStepSize, self.numMolec))
        yOffset = np.asarray(yOffset)

        

        if self.twoDiffusion == True:
            xOffset2 = positionGenerator(self.secondDiffusion, timeStepSize, self.numMolec)
            xOffset2 = np.asarray(xOffset2)
            yOffset2 = (positionGenerator(self.secondDiffusion, timeStepSize, self.numMolec))
            yOffset2 = np.asarray(yOffset2)

            #Now apply it to our original xOffset
            lower = self.xbound - 0.5 
            upper = self.xbound + 0.5 

            for i in range(len(self.xPositions)):
                curPos = self.xPositions[i]
                if curPos < lower:
                    continue
                if curPos > upper:
                    xOffset[i] = xOffset2[i]
                if curPos > lower and curPos < upper:
                    fracAcross = (curPos - lower)/(upper - lower)   #how far across the boundary is it
                    DinBetween = self.diffusionCoefficient*(1-fracAcross) + self.secondDiffusion*(fracAcross)
                    #Generate a step 
                    step = positionGenerator(DinBetween,timeStepSize,1)
                    xOffset[i] = step 
            
        #Now time to see if we need to check
        xAccept = np.ones((self.numMolec))       #Intiially start by accepting all
        yAccept = xAccept.copy()

        #CYTOSKELETON CONFINEMENTS
        if self.skeleton == True:
           

            self.xPositions,        #List of N particles
            self.yPositions         #List of N particles

            positions = [[i,j] for i,j in zip(self.xPositions,self.yPositions)]
            offset = [[i,j] for i,j in zip(xOffset,yOffset)]


            newPositions = skeletonJumper(positions,offset,self.confinements)
           
            newPositions = np.array(newPositions)

            self.xPositions = newPositions[:,0]; self.yPositions = newPositions[:,1]
        




        #CONFINEMENTS
        if self.skeleton == True:
              pass

        if self.skeleton == False:
            self.xPositions = self.xPositions+xOffset
            self.yPositions = self.yPositions+yOffset
        if self.periodic == True:
            
            
            
            self.xPositions = self.xPositions%self.ROI
            self.yPositions = self.yPositions%self.ROI
            return
        
        return 
    #Return a bunch of diffusion
    def simulate(self,timeSteps,timeStepSize):
        '''
        Simulates our cells
        '''
        #Reset our position history
        self.xPosHist = []
        self.yPosHist = []
        returnedArray = []
        
        #Make our molecules blink
        if self.blinking == True:
            measure, times = self.gSim(self.rateOn,self.rateOff,timeSteps*timeStepSize)


            #Now we want to randomly choose molecules to turn on and off
            
            gillIndex = 0
        for i in range(timeSteps):
            if self.blinking == True:
                #Gillespie time
                curTime = i*timeStepSize

                while times[gillIndex] < curTime:
                    gillIndex += 1
                gillIndex -= 1
                if curTime == 0:
                    gillIndex = 0

            #Now that we have the proper index we need to look at 

                #If we have too many
                while measure[gillIndex][0] < sum(self.onMolecules):
                    randomIndex = np.random.randint(self.numMolec)
                    if self.onMolecules[randomIndex] == 1:
                        self.onMolecules[randomIndex] = 0
            

                #If we have too few
                while measure[gillIndex][0] > sum(self.onMolecules):
                    randomIndex = np.random.randint(self.numMolec)
                    if self.onMolecules[randomIndex] == 0:
                        self.onMolecules[randomIndex] = 1
            
            
        
            returnedArray.append(self.imageExport())
            self.xPosHist.append(self.xPositions.copy())
            self.yPosHist.append(self.yPositions.copy())
            self.diffuse(timeStepSize)
            self.floureHist.append(sum(self.onMolecules))        #FOR DIAGNOSTICS
            
        
        means = []
        for i in returnedArray:
            means.append(np.max(i))
        std = np.mean(means)/self.S2N
        newArray = []
        for image in returnedArray:
            if self.S2N !=0:
                noise = np.random.normal(loc = 0.0, scale = std,size = (np.shape(image)[0],np.shape(image)[0]))

            #If we have noise  
            else:
                noise = np.zeros((np.shape(image)[0],np.shape(image)[0]))
            image = (image+noise)
            if np.min(image) < 0:
                image = image + abs(np.min(image))
            newArray.append(image)
            #Remove all the <0



        return newArray

    #Defining a region with a different diffusion coefficient
    def diffusionRegion(self,xbound,newDiffusion):
        '''For our diffusion map 
        xBound = our barrier, if a molecule is in the region greater than that it'll diffuse with the second diffusion coefficient
            '''
        self.secondDiffusion = newDiffusion
        self.xbound = xbound 
        self.twoDiffusion = True 

    #Defining our Gillespie Simulation





#===== Helper functions =====

def crossingChecker(skeleton,proposedJump):
    '''
    Given the skeleton line and the proposedJump, check if it crosses
    '''
    #Find the intersection points

    #Defining functions only used here
    def onLine(line,point):
    #Determine if the point is on the parameterized line
        x,y = point 
        
        if (0< (x-line.x1)/(line.x2 - line.x1) <1) or (0 <(y-line.y1)/(line.y2  - line.y1) <1) :

            return True 
        else:
            return False

    def intersectionPoints(skeleton,proposedJump):
        x1,y1,x2,y2 = skeleton.returnPoints(); x3,y3,x4,y4 = proposedJump.returnPoints()
        
        #So there is a chance that it'll be parallel
        
        
        numeratorX =((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))
        numeratorY = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))
        denominator = ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        if denominator == 0:        #The are parallel or the same line
            return(np.inf,np.inf)

        px = numeratorX/denominator; py = numeratorY/denominator


        return px,py

    px,py = intersectionPoints(skeleton,proposedJump)
    if px == np.inf or py == np.inf:
        return True
    #Both on Line and we fail to jump
    if onLine(skeleton,(px,py)) and onLine(proposedJump,(px,py)) and not (np.random.rand() < skeleton.p):
        return False 
    return True

      #Vectorize our crossing Checker so we can pass the array

def skeletonJumper(positions,offset,completeSkeleton):
    '''
    We now check if our molecules have crossed ANY of the lines
    Params:
        positions Nx2: x1,y1, N molecules: The original positions of our molecules 
        offset Nx2:x2,y2    The proposed jump of our molecules
        completeSkeleton:   All of the line objects
    Returns:
        positions, updated with the new coordinates if they made the jump
    '''
    
    
    proposed = np.array(positions).T + np.array(offset).T                            #Where the molecules moy go
    proposed = proposed.T
   
    
    jumpLines = [[i[0],i[1],j[0],j[1]] for i,j in zip(positions,proposed)]          #Creates a Nx4 array, x1,y1,x2,y2

    moleculeJumps = [makeLine(i) for i in jumpLines]        #Our molecules jumps
    #Now we see what we accepts
    
    returnedList = []

    def crossingListChecker(moleculeJump): return [crossingChecker(i,moleculeJump) for i in completeSkeleton]
    returnedList = [crossingListChecker(i) for i in moleculeJumps]
    
             
    #Take accepted jumps, set new positions
    

    #Not quite, so what we have is now a list of line checks
    acceptedArray = [not all( i) == False for i in returnedList]
    
    #Oh it's weird... I keep setting it to be sillt
    returnedArray = []
    for i,j,z in zip(acceptedArray,positions,proposed):
        if i == True:
            returnedArray.append(z)
            continue 
        returnedArray.append(j)

               #Where we can go, we go
    return returnedArray                                        #Returns the new jump
    
#our testing code 
if __name__ == '__main__' : 
    #New confinements

    #TEST- Initializing skeleton lines
    simulationObject = molecule()       #Simulation object

    skeletonLines = [
        [0.2,0.2,1.1,1.1],
        [0,-10,0,10],
        [4.1,4.4,-8.5,-8.6],
        [10,10,-10,-10,1]
        ]                              #Our test skeleton lines
    simulationObject.confinementInitializer(skeletonLines)
    print(simulationObject.confinements)
    
    positions = np.array([[-1,0],[1,1],[5,5]])
    proposedPositions = np.array([[1,0],[2,2],[3,3]])


    print(skeletonJumper(positions,proposedPositions,simulationObject.confinements))

    


        
        

    





