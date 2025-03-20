import numpy as np 
import scipy
import time 
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

    def returnPoints(self):
        '''Returns the points which define the line'''
        return self.x1,self.y1,self.x2,self.y2
def distance(line1,line2,s,t):
    '''
    returns the distance between two lines
    '''
    return np.sum((line1(s) - line2(t))**2)
def intersectionChecker(line1,line2):
    '''Given two lines.... do they intersect'''
    toMinimize = lambda s: distance(line1,line2,s[0],s[1])  #Define our minimization function
    minimization = (scipy.optimize.minimize(toMinimize,[0.5,0.5],bounds = [(0,1),(0,1)]))   #minimize this
    #if the molecule crosses the barrier return true, else return false
    if minimization['fun'] < 1e-15:
        return True 
    else:  
        return False





def intersectionPoints(line1,line2):
    x1,y1,x2,y2 = line1.returnPoints(); x3,y3,x4,y4 = line2.returnPoints()
    print('hello')
    #So there is a chance that it'll be parallel
    try:
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    except:
        print('Lines are Parallel')
        return (0,0)
    return px,py



myline = line(-1,1,1,-1,0); myline2 = line(-1,-1,-4,-4)
func = myline.vectorize(); func2= myline2.vectorize()
print(func(0.5))

print(intersectionPoints(myline,myline2))


#Now we determine if the distance functional can be properly minimized

toMinimize = lambda s: distance(func,func2,s[0],s[1])

x0 = [0,0]
minimization = (scipy.optimize.minimize(toMinimize,x0))
time1 = time.time()
print(intersectionChecker(func,func2))
print(time.time() - time1)
#Test Case
#Line1 (-1,1) -> (1,-1)
#Line2 (-1,-1) -> (1,1)