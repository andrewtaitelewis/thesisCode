import numpy as np 
import matplotlib.pyplot as plt 
import moleculeSimulation as molec
import helper
import matplotlib.pyplot as plt 
import scipy
import numpy as np
from waveletAnalysis import *
import waveletTransformation as WT
from Correlations import *





if __name__ == "__main__":
   
    simulationObject = molec.molecule(numMolec= int(1e4),xVelocity = 1,imageResolution=128,periodic= True)
    imageSeries = simulationObject.simulate(300,0.02)

    helper.timeSeriesVisualizer(imageSeries,0.01)
    #Now what would a correlation function look like?
    transformation = WT.waveletTransform2d(imageSeries,KernelCreatorRicker2d(simulationObject,0.4,10),periodic=False)
    
    corrFunc = ticsFFT(imageSeries)
    plt.plot(corrFunc[0])
    plt.show()
