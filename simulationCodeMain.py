#Some imports 
import moleculeSimulation as ms 
import numpy as np 
import time

#====== Main Code =====


simulationObject = ms.molecule(1000,ROI=11.7185,imageResolution=150)

#We need to set out confinements
#0.5 um around the edges.
#1um on average in the top left

#Confinements:

#Takes 4 and a bit minutes...
def confinementMakerGrid(xs,ys,p):
    '''Having two lists, make the grid'''
    xMin = xs[0]; xMax = xs[-1]
    yMin = ys[0]; yMax = ys[-1]
    toInitialize = []

    for i in xs:
        toInitialize.append([i,yMin,i,yMax,p])
    for j in ys:
        toInitialize.append([xMin,j,xMax,j,p])
    return toInitialize

#quadrant 1 Hop Diffusion
xs = np.arange(0,5,0.5) + 0.85925
ys = np.arange(0,5,0.5) + 0.85925
simulationObject.confinementInitializer(confinementMakerGrid(xs,ys,0.01))
#Quadrant 2 Caged Diffusion 1um
xs = np.arange(6,10,1) + 0.85925; ys = np.arange(0,5,1) + 0.85925
simulationObject.confinementInitializer(confinementMakerGrid(xs,ys,0)) 
#Quadrant 3: Hop Diffusion, 1um
xs = np.arange(6,10,1) + 0.85925; ys = np.arange(0,5,1) + 0.85925
simulationObject.confinementInitializer(confinementMakerGrid(xs,ys,0.1))



time1 = time.time()
Images = simulationObject.simulate(1000,0.02)

Images = np.array(Images)[:,11:139,11:139]

print(time.time()-time1)
np.save('confinement',Images)

#=======
simulationObject = ms.molecule(1500)
simulationObject.confinementInitializer([[0,10,0,-10],[-10,0,10,0]])

time1 = time.time()
Images = simulationObject.simulate(100,0.02)

print(time.time()-time1)
np.save('test',Images)

