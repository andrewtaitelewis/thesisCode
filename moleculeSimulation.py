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



        self.confinements



    #Sub Class for the confinements
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
    
    


#Now given they intersect, are they both within boundaries


def crossingChecker(line1,line2):
    #Find the intersection points

    #Defining functions only used here
    def onLine(line,point):
    #Determine if the point is on the parameterized line
        x,y = point 
        if 0< (x-line.x1)/(line.x2 - line.x1) <1:
            return True 
        else:
            return False

    def intersectionPoints(line1,line2):
        x1,y1,x2,y2 = line1.returnPoints(); x3,y3,x4,y4 = line2.returnPoints()
        
        #So there is a chance that it'll be parallel
        try:
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        except:
            print('Lines are Parallel')
            return (0,0)
        return px,py

    px,py = intersectionPoints(line1,line2)
    #See if they're both on the line
    if onLine(line1,(px,py)) and onLine(line2,(px,py)):
        return True 
    else:
        return False

#Cytoskeleteon confinement
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

    #If we have cytoskeletal confinements
    if self.skeleton == True:
        #Go through particles- see if any jump over a line
        #If they do jump over a line- see if they can (probability)
        #if True- jump, if False- we'll need to reflect it
        #Check the x 
        xAccepted,xFinalPositions = skeletonJumper(self.xPositions,xOffset,self.xSkeleton,self.jumpProb)
        yAccepted,yFinalPositions = skeletonJumper(self.yPositions,yOffset,self.ySkeleton,self.jumpProb)
        xAccept= xAccept*xAccepted
        yAccept = yAccept*yAccepted




    #If we have lipid rafts
    #For each pair of positions we see if it has entered any lipid domain
    if self.lipidRaft == True:
        for i in range(self.numMolec):
            xPos,yPos = self.xPositions[i],self.yPositions[i]
            xPro,yPro = xPos + xOffset[i], yPos + yOffset[i]
            #Over all our rafts
            for j in range(len(self.lipidRaftCenters)):
                Accepted = lipidDomainCrosser(self.lipidRaftCenters[j], self.lipidRaftRadius[j], 
                (xPos,yPos), (xPro,yPro), self.lipidRaftJumpProb[j])
                xAccept[i] = xAccept[i]*Accepted
                yAccept[i] = yAccept[i]*Accepted

    if self.skeleton == True:
            
            self.xPositions = xFinalPositions
            self.yPositions = yFinalPositions

    if self.skeleton == False:
        self.xPositions = self.xPositions+xOffset
        self.yPositions = self.yPositions+yOffset
    if self.periodic == True:
        
        
        
        self.xPositions = self.xPositions%self.ROI
        self.yPositions = self.yPositions%self.ROI
        return
    else:       #not periodic
        print('wtf')
        self.xPositions += (xOffset*xAccept)
        self.yPositions += (yOffset*yAccept)
        if self.skeleton == True:
            
            self.xPositions += int(xAccept == 0)*(xWorstCasePos-self.xPositions)
            self.xPositions += int(yAccept == 0)*(yWorstCasePos-self.yPositions)
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




#Helper functions





def cytoskeletonCrosser(skeleton,curPos,proPos,jumpProb):
    ''' 
    Given, currentPosition, proposedPosition and the skeleton check whether the walker crosses the skeleton \n
    Params:\n-------------
    skeleton, [floats]: the array of skeleton positions
    curPos, float: The current coordinate (in 1d) of the molecule
    proPos, float: The propoesed position to jump to of the molecule
    jumpProb, float: between 1 and 0, probability strength of confinement
    Returns: \n-------------
    Boolean of whether the molecule has crossed a skeleton or not
    Float: final position, whether it was jumped or not
    '''
    #Formatting- Make sure everything is in the spot we want 
    skeleton = np.array(skeleton)

    test1 = skeleton - curPos 
    test2 = skeleton - proPos   #If when we multiply these together we get a negative value we will have crossed
    
    if any(x < 0 for x in np.multiply(test1,test2)):
        array = list(np.multiply(test1,test2) < 0)
        index = 0 
        testArray = skeleton -curPos 
        while testArray[index] < 0:
            index += 1
        index = index - 1
        if proPos > curPos:
            index +=1
        
         
        skeletonPosition = skeleton[index]  #The position in um 
        

        #reflect it off the boundary   
        #Now time to see if it'll jump 
 
        if np.random.random() < jumpProb:
            #If it jumps
            return True, proPos
        gridSize = (skeleton[index]-skeleton[index-1])
        if skeletonPosition < curPos:
            endPos = skeletonPosition + abs(proPos-curPos)%gridSize

        elif skeletonPosition > curPos:
            endPos = skeletonPosition - abs(proPos-curPos)%gridSize
        return False,endPos
    else:   
        #doesn't hit anything and jumps
        return False, proPos



def skeletonJumper(positions,offset,skeleton,jumpProb):
    '''
    
    Checks whether or not a molecule can jump over a cytoskeletal element:
    Params:
    positions, [float]: where the particles are pre jump
    offset, [float]: the proposed offsets of the particles
    skeleton, [float]: the location of the skeletal elements
    jumpProb, float[0,1]: the probability a molecule can jump over a cytoskeletal element
    Returns
    acceptArray: [bool]: the list of whether or not a jump has been accepted
    newPositions: The worst case positions it could end up in
    
    '''
    acceptArray = np.ones(len(positions))
    newPositions = np.zeros(len(positions))
    #Check the x 
    proposed = positions+offset
    counter = -1


    for i,j in zip(positions,proposed):
        counter += 1
        
        crossedBool,endPos = cytoskeletonCrosser(skeleton,i,j,jumpProb) #Did it cross a boundary
        
        newPositions[counter] = endPos 
    
    


    #I am going to assume that if the jump is rejected it'll 'bounce' off 

    
    return np.array(acceptArray),newPositions

#Determines whether or not a walker has crossed a lipid domain
def lipidDomainCrosser(location, radii,currentPosition,proposedPosition,probability):
    '''
    Checks to see if a walker has crossed a lipid domain, one dimensional, so one would \n
    check the x coordinate and the y coordinate seperately.
    Params:
    -------
    - locations: list:floats: a list of the centers of the lipid microdomains
    - radii: list:floats: a list of the radius' of the lipid microdomains
    - currentPosition: (float,float): where the walker currently is
    - proposedPosition: (float,float): where the walker wants to go
    - probability: float, 0<1: probaiblity that the walker will cross the membrane
    Returns:
    - boolean: Whether or not the jump was successful 
    '''
    #Checks to see if a point is in a circle
    def cirleChecker(position,location,radii):
        '''Small checker to see if a point is in a circle'''
        xCent,yCent = location 
        xPos, yPos = position
        if (xPos - xCent)**2 + (yPos - yCent)**2 <= radii**2:
            return True 
        else:
            return False 
    
    #If current and proposed positions are in the domain continue
    curPosBool = cirleChecker(currentPosition, location, radii)
    proPosBool = cirleChecker(proposedPosition, location, radii)
    if  curPosBool and proPosBool or (not curPosBool) and (not proPosBool):
        return True #i.e jump is accepted because of the check
    elif np.random.uniform() < probability:
        return True 
    else: 
        return False  
     



#our testing code 
if __name__ == '__main__' : 
    #Cytoskeleton crosser test
    cytoarray = np.array([1,2,3,4,5])
    prePos = 2.5
    postPos = 2.8 
    print(cytoskeletonCrosser(cytoarray,prePos,postPos))




        
        

    





