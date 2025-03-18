import numpy as np 
import scipy
import scipy.optimize 

#Trying to do a 

class line:

    def __init__(self,x1,y1,x2,y2,p = 0):
        '''
        Defines our line
        '''
        self.x1 = x1; self.y1 = y1  #Coordinates of one end point of the line
        self.x2 = x2; self.y2 = y2  #Coordinates of the second end point of the line
        self.p = p                  #Probability for a molecule to jump over the line
        pass



    def vectorize(self):
        '''
        Makes a vector form of our confinement line where 0<s<1
        '''
        return lambda s: np.asarray([(self.x2-self.x1)*s +self.x1, (self.y2-self.y1)*s + self.y1])

def distance(line1,line2,s,t):
    '''
    returns the distance between two lines
    '''
    return np.sum((line1(s) - line2(t))**2)
def intersectionChecker(line1,line2):
    '''Given two lines.... do they intersect'''




myline = line(-1,1,1,-1,0); myline2 = line(-1,-1,1,1)
func = myline.vectorize(); func2= myline2.vectorize()
print(func(0.5))

print(distance(func,func2,0.5,0.5))


#Now we determine if the distance functional can be properly minimized

toMinimize = lambda s: distance(func,func2,s[0],s[1])

x0 = [0,0]
minimization = (scipy.optimize.minimize(toMinimize,x0))
print(minimization['fun'] < 1e-15)
#Test Case
#Line1 (-1,1) -> (1,-1)
#Line2 (-1,-1) -> (1,1)