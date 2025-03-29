import time 
import waveletTransformation as WT 
import waveletAnalysis as WA 
import moleculeSimulation as mc
import tifffile
import sys  #We could do argpause instead....
import numpy as np #A classic
import pandas as pd
import scipy
import multiprocessing
import pickle


def model(t,td,D,L,a,w0,sigma):
    '''Catch all hop diffusion model'''
    return a/(((L**2)/3)*(1-np.exp(-t/td))+ 4*D*t + w0**2 +4*sigma**2)**3

 
