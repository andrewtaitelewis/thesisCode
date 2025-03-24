#Some imports 
import moleculeSimulation as ms 
import numpy as np 
import time

#====== Main Code =====

simulationObject = ms.molecule(1500)
simulationObject.confinementInitializer([[0,10,0,-10],[-10,0,10,0]])

time1 = time.time()
Images = simulationObject.simulate(100,0.02)

print(time.time()-time1)
np.save('test',Images)