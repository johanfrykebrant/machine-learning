import numpy as np 
from numpy import dot,shape,linalg
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 

"""
https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf
"""

def create_test_data():
    n_samples = 100
    n_features = 1
    noise = 25
    bias = random.uniform(-100, 100)
    # X = features, Y = target variable
    X,Y = make_regression(n_samples = n_samples,
                        n_features = n_features,
                        noise = noise,
                        bias = bias)
    # Dividing data set into training and testing data sub sets
    X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size=0.1)

    return X_tr,X_te,Y_tr,Y_te


class LinearRegression:
    def __init(self,x):
        #Add row of ones, this is refered to as the intercept term.
        x = np.c_[np.ones(shape(x)[0]),x]
        return x

    def fit(self,x,y):
        x = self.__init(x)
        # a @ b is the same as dot(a,b) using it here to make it easier to read
        self.theta = (linalg.inv((x.T @ x)) @ x.T ) @ y

        return self.theta
    
    def cost(self,x,y):
        x = self.__init(x)
        # z is the estimated value of y
        z = dot(x,self.theta)
        # reshape y & z so that matrix mulitplications will work properly
        y = np.reshape(y,(len(y),1))
        z = np.reshape(z,(len(z),1))
        # diff is the differance between the estimated and actual value of y
        diff = z - y
        # J is the cost function defined as the sum of the squared differnances  
        J = dot( diff.T,diff )
        return round(J[0][0],2)

if __name__ == "__main__":
    X_tr,X_te,Y_tr,Y_te = create_test_data()
    linReg = LinearRegression()
    t = linReg.fit(X_tr,Y_tr)
    print("Cost for testing dataset: {}".format(linReg.cost(X_te,Y_te)))
    x = np.linspace(-3,3,100)
    y = t[0] + t[1]*x
        
    plt.figure()
    plt.scatter(X_tr,Y_tr,zorder=2.0,s=16, label = "raw data")
    plt.plot(x,y,zorder=1.0,c='r',label = "fitted line")
    plt.legend()
    plt.grid(zorder=-1.0)
    plt.show()

