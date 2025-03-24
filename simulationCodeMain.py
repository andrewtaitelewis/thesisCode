#Some imports 
import moleculeSimulation as ms 


#====== Main Code =====

simulationObject = ms.molecule(1000)
simulationObject.confinementInitializer([[0,10,0,-10]])

Images = simulationObject.simulate(300,0.02)