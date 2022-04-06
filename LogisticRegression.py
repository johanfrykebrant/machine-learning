import numpy as np 
from numpy import log,dot,e,shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 

""""
The following pdf from Stanford University covers the theory of Ligistic Regression well. I used that as a basis for this implementation:

https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf

Link below decribes the cost function (Log likelihood) in a bit more detail:

https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148#:~:text=For logistic regression%2C the Cost,Graph of logistic regression

Link below describes the theory behind the scoring method(F1 Score):

https://towardsdatascience.com/the-f1-score-bec2bbc38aa6

This example will only cover logic regression with one feature and two classes for simplicity. The code will differ for a multi variable version of Logistic regression but the principle is the same.
"""

def create_test_data():
    # Create data set for classification
    n_samples = 1000
    n_features =1
    n_classes=2
    X,Y = make_classification(n_samples = n_samples,
                            n_features =n_features,
                            n_classes=n_classes,
                            n_informative=1,
                            n_redundant=0,
                            n_clusters_per_class=1)

    # Dividing data set inte training and testing data sets
    X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size=0.1)

    return X_tr,X_te,Y_tr,Y_te

class LogisticRegression:
    # Intialize helper functions
    def __sigmoid_func(self,z):
        """This function will accept a vector as input an variable"""
        return 1/(1+e**(-z))

    def __cost_gradient(self,x,y):
        """ returns the gradient/derivative of the cost function""" 
        z = dot(x,self.theta)
        # Need to reshape Y for vector subtraction to work propperly
        #       dot(X.T,  self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
        
        return dot(x.T,( self.__sigmoid_func(z) - np.reshape(y,(len(y),1)) ))

    def __init(self,X):
        """ Initialize the theta and features vectors """
        # Standardize data set, this is always good practice.
        X = (X - np.mean(X))/np.std(X)

        # For every row in the X vector(lets call it x), we want to be able to perform dot(x,theta) such that dot(x,theta) = theta0 + Xtheta1
        # Therefore, a new column of ones need to be addede to the X vector.
        X = np.c_[np.ones(shape(X)[0]),X]
        return X

    def train(self,X,Y,eta=0.001,itr=1000):
        X = self.__init(X)
        self.theta = np.zeros((shape(X)[1],1))

        def cost():
            z = dot(X,self.theta)
            c0 = dot( Y.T , log(self.__sigmoid_func(z)) )
            c1 = dot( (1-Y).T , log(1 - self.__sigmoid_func(z)))
            return -(c0 + c1)/len(Y)
        
        cost_list = np.zeros(itr,)
        for i in range(itr):
                self.theta = self.theta - eta * self.__cost_gradient(X,Y)
                cost_list[i] = cost()
            
        print("theta = [{},{}]".format(self.theta[0],self.theta[1]))
        return cost_list
    
    def score(self,y_pred,y_true):
        """Score the model"""
        true_pos,true_neg,false_pos,false_neg = 0,0,0,0
        for i in range(len(y_pred)):
            if y_true[i]==1 and y_pred[i]==1:
                true_pos += 1
            if y_true[i]==1 and y_pred[i]==0:
                false_neg += 1
            if y_true[i]==0 and y_pred[i]==0:
                true_neg += 1
            if y_true[i]==0 and y_pred[i]==1:
                false_neg += 1
        # true possitives/(true possitives + false possitives)
        precission = true_pos/(true_pos + false_pos)
        # true possitives/(true possitives + false negatives)
        recall = true_pos / (true_pos + false_neg)
        return 2*(recall*precission)/(recall + precission)

    def predict(self,x):
        """return aprediction of based on feature values"""
        x = self.__init(x)
        z = dot(x,self.theta)
        pred = []
        for i in self.__sigmoid_func(z):
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)

        return pred

if __name__ == "__main__":
    X_tr,X_te,Y_tr,Y_te = create_test_data()

    logReg = LogisticRegression()
    costs = logReg.train(X_tr,Y_tr)
    prediction = logReg.predict(X_te)
    score = logReg.score(prediction,Y_te)
    print("Score for testing dataset: {}".format(score))

    fig, axs = plt.subplots(1, 3)
    axs[0].scatter(np.concatenate((X_tr, X_te), axis=None),np.concatenate((Y_tr, Y_te), axis=None),s=8)
    axs[0].set(xlabel='feature value', ylabel='class')
    axs[0].set_yticks([0,1])

    axs[1].plot(costs)
    axs[1].set(xlabel='iterations', ylabel='cost')

    axs[2].scatter(X_te,prediction,c='b',label="prediction",s=32,alpha=0.3)
    axs[2].scatter(X_te,Y_te,c='k',label='actual',s=4)
    axs[2].legend()
    axs[2].set(xlabel='feature value', ylabel='class')
    axs[2].set_yticks([0,1])

    prediction = logReg.predict(X_tr)
    score = logReg.score(prediction,Y_tr)
    print("Score for training dataset: {}".format(score))

    plt.autoscale()
    plt.show()
